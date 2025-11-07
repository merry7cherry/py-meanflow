# Mean Flows: PyTorch + GPU Implementation

<div align="center">
<img width="800" alt="Image" src="https://github.com/user-attachments/assets/2adc06a5-c3bf-41c8-acfa-54c822e7c07b" />
</div>


This is a PyTorch+GPU re-implementation for the CIFAR-10 experiments in [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447). The original experiments were done in JAX+TPU.

## Installation

This repo was tested in PyTorch 2.7.1 and uses `torch.compile`. Compilation may depend on PyTorch versions.

```
conda env create -f environment.yml
conda activate meanflow
```

## Demo

Run `demo.ipynb` for a demo of 1-step generation and FID evaluation. This demo should produce <2.9 FID.

<div align="center">
<img width="480" alt="Image" src="https://github.com/user-attachments/assets/11966c45-25c6-44e5-ae24-75b29e697b9b" />
</div>


## Training

Launch training directly with Python on a single GPU:

```
python -m meanflow.train \
    --output_dir=./tmp \
    --dataset=cifar10 \
    --batch_size=128 \
    --lr=0.0006 \
    --eval_frequency=50 \
    --epochs=16000 \
    --compute_fid \
    --log_per_step=100 \
    --tr_sampler=v1 \
    --P_mean_t -0.6 \
    --P_std_t 1.6 \
    --P_mean_r -4.0 \
    --P_std_r 1.6 \
    --warmup_epochs 200 \
    --norm_p 0.75 \
    --ratio 0.75 \
    --dropout 0.2 \
    --use_edm_aug
```

The helper scripts in `meanflow/scripts` now mirror this single-GPU command without requiring `torchrun`.


## Note on JVP

Users may be unfamiliar with the JVP (Jacobian-vector product) operation, which MeanFlow is based on. While JVP is straightforward to implement in JAX, its correct implementation in PyTorch is worth a closer look.

#### Compilation

The memory and speed of JVP can greatly benefit from compilation, in both JAX and PyTorch. In our code, this is done by:
```
compiled_train_step = torch.compile(
    train_step,
    disable=not args.compile,
)
```
where `train_step` is:
```
def train_step(model, *args, **kwargs):
    loss = model.forward_with_loss(*args, **kwargs)
    loss.backward(create_graph=False)
    return loss
```
Optionally, we also put `update_ema()` into `train_step` for compilation.

#### Alternative to Compilation

If you don't want to compile (for example, some of your ops are not supported), we recommend to compute `dudt` by `torch.func.jvp` under `torch.no_grad()`:
```
u_pred = u_func(z, t, r)
with torch.no_grad():
    _, dudt = torch.func.jvp(u_func, (z, t, r), (v, dtdt, drdt))
```
The function prediction `u_pred` is computed separately. In this way, computing `dudt` does not introduce substantial additional memory usage, and its time cost is roughly equivalent to a forward and backward pass. If you want `u_func` to share the dropout masks, consider backing up rng states by `cpu_rng_state = torch.get_rng_state(); cuda_rng_state = torch.cuda.get_rng_state()` and restoring by `torch.set_rng_state(cpu_rng_state); torch.cuda.set_rng_state(cuda_rng_state)` before and after the call of `u_func`.

## References

This repo is based on the following repos:

* [Flow Matching repo](https://github.com/facebookresearch/flow_matching)
* [EDM repo](https://github.com/NVlabs/edm)

See also:

* [Our MeanFlow JAX repo](https://github.com/Gsunshine/meanflow) with ImageNet experiments.
* [A third-party MeanFlow PyTorch repo](https://github.com/zhuyu-cs/MeanFlow) with reproduced ImageNet results.
