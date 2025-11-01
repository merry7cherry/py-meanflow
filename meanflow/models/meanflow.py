import torch

import torch.nn as nn

from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net
from models.vae import LightVAE


class MeanFlow(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(MeanFlow, self).__init__()
        self.net = arch(**net_configs)
        self.args = args

        self.latent_dim = getattr(self.net, "latent_dim", 0)
        self.m_encoder = (
            LightVAE(
                img_resolution=net_configs["img_resolution"],
                in_channels=net_configs["in_channels"],
                latent_dim=self.latent_dim,
                augment_dim=net_configs.get("augment_dim", 0),
            )
            if self.latent_dim > 0
            else None
        )

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
        if self.m_encoder is not None:
            eps = torch.randn(x.shape[0], self.latent_dim, device=device, dtype=x.dtype)

            time_steps = (t, t - r)
            mu, logvar = self.m_encoder.encode(e, x, z, time_steps, aug_cond)
            latent_m = self.m_encoder.reparameterize(mu, logvar, eps)
            kl_loss = self.m_encoder.kl_divergence(mu, logvar).mean()

            def m_func(e_in, x_in, z_in, t_in, r_in):
                mu_in, logvar_in = self.m_encoder.encode(
                    e_in,
                    x_in,
                    z_in,
                    (t_in, t_in - r_in),
                    aug_cond,
                )
                return self.m_encoder.reparameterize(mu_in, logvar_in, eps)

            _, dmdt = torch.func.jvp(
                m_func,
                (e, x, z, t, r),
                (
                    torch.zeros_like(e),
                    torch.zeros_like(x),
                    v,
                    torch.ones_like(t),
                    torch.zeros_like(r),
                ),
            )

            def u_func(z_in, t_in, r_in, m_in):
                h_in = t_in - r_in
                return self.net(z_in, (t_in.view(-1), h_in.view(-1)), aug_cond, latent=m_in)

        else:
            latent_m = None
            dmdt = None
            kl_loss = 0.0

            def u_func(z_in, t_in, r_in):
                h_in = t_in - r_in
                return self.net(z_in, (t_in.view(-1), h_in.view(-1)), aug_cond)

        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)

        with torch.amp.autocast("cuda", enabled=False):
            if self.m_encoder is not None:
                u_pred, dudt = torch.func.jvp(
                    u_func,
                    (z, t, r, latent_m),
                    (v, dtdt, drdt, dmdt),
                )
            else:
                u_pred, dudt = torch.func.jvp(
                    u_func,
                    (z, t, r),
                    (v, dtdt, drdt),
                )

            u_tgt = (v - (t - r) * dudt).detach()

            loss = (u_pred - u_tgt)**2
            loss = loss.sum(dim=(1, 2, 3))  # squared l2 loss
            
            # adaptive weighting
            adp_wt = (loss.detach() + self.args.norm_eps) ** self.args.norm_p
            loss = loss / adp_wt

            loss = loss.mean()  # mean over batch dimension

            if self.m_encoder is not None:
                loss = loss + kl_loss

        return loss
    
    def sample(self, samples_shape, net=None, device=None):
        net = net if net is not None else self.net_ema                

        e = torch.randn(samples_shape, dtype=torch.float32, device=device)
        z_1 = e
        t = torch.ones(samples_shape[0], device=device)
        r = torch.zeros(samples_shape[0], device=device)
        latent = None
        if self.m_encoder is not None:
            latent = torch.randn(
                z_1.shape[0], self.latent_dim, device=device, dtype=z_1.dtype
            )

        u = net(z_1, (t, t - r), aug_cond=None, latent=latent)
        z_0 = z_1 - u
        return z_0
