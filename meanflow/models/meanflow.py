import torch

import torch.nn as nn

from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net


class MeanFlow(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(MeanFlow, self).__init__()
        self.net = arch(**net_configs)
        self.args = args

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))
        
        self.net_ema = init_ema(self.net, arch(**net_configs), args.ema_decay)

        # maintain extra ema nets
        self.ema_decays = args.ema_decays
        for i, ema_decay in enumerate(self.ema_decays):
            self.add_module(f"net_ema{i + 1}", init_ema(self.net, arch(**net_configs), ema_decay))

    def update_ema(self):
        self.num_updates += 1
        # num_updates = self.num_updates.item()
        num_updates = self.num_updates

        update_ema_net(self.net, self.net_ema, num_updates)

        # update extra ema
        for i in range(len(self.ema_decays)):
            update_ema_net(self.net, self._modules[f"net_ema{i + 1}"], num_updates)

    def forward_with_loss(self, x, aug_cond):

        device = x.device
        e = torch.randn_like(x).to(device)
        t, r = sample_two_timesteps(self.args, num_samples=x.shape[0], device=device)
        t, r = t.view(-1, 1, 1, 1), r.view(-1, 1, 1, 1)

        z = (1 - t) * x + t * e
        v = e - x

        # define network function
        def u_func(z, t, r):
            h = t - r
            return self.net(z, (t.view(-1), h.view(-1)), aug_cond)

        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)

        with torch.amp.autocast("cuda", enabled=False):
            u_pred, dudt = torch.func.jvp(u_func, (z, t, r), (v, dtdt, drdt))
        
            u_tgt = (v - (t - r) * dudt).detach()

            loss = (u_pred - u_tgt)**2
            loss = loss.sum(dim=(1, 2, 3))  # squared l2 loss
            
            # adaptive weighting
            adp_wt = (loss.detach() + self.args.norm_eps) ** self.args.norm_p
            loss = loss / adp_wt

            loss = loss.mean()  # mean over batch dimension
        
        return loss
    
    def sample(
        self,
        samples_shape,
        net=None,
        device=None,
        steps: int = 1,
        *,
        target_oriented: bool = True,
    ):
        """Draw samples by integrating the mean-flow velocity field.

        Args:
            samples_shape: Shape of the samples to generate.
            net: Network used for inference. Defaults to the EMA network.
            device: Device used for generation.
            steps: Number of uniform inference steps between time 0 and 1.
            target_oriented: Whether to use target-oriented stepping that
                mirrors the training reference schedule. When ``False`` the
                reference time is fixed to zero for every step.

        Returns:
            Tensor of generated samples corresponding to time 0.
        """

        if steps < 1:
            raise ValueError("steps must be a positive integer")

        net = net if net is not None else self.net_ema
        device = device if device is not None else next(net.parameters()).device

        was_training = net.training
        net.eval()

        with torch.no_grad():
            current = torch.randn(samples_shape, dtype=torch.float32, device=device)
            batch_size = samples_shape[0]
            times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=current.dtype)

            for idx in range(steps):
                current_time = times[idx]
                next_time = times[idx + 1]
                evaluation_time = 1.0 - current_time
                if target_oriented:
                    reference_time = 1.0 - next_time
                else:
                    reference_time = 0.0

                t = torch.full((batch_size,), evaluation_time, device=device, dtype=current.dtype)
                r = torch.full((batch_size,), reference_time, device=device, dtype=current.dtype)
                h = t - r

                velocity = net(current, (t, h), aug_cond=None)
                dt = next_time - current_time
                current = current - velocity * dt

        if was_training:
            net.train(True)

        return current
