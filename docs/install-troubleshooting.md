# Install troubleshooting

Install JAX for your GPU following the
[JAX instructions](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu),
then check that it is seen:

```bash
python -c "import jax; print(jax.devices())"
```

This should list a `CudaDevice`. The warning "a CUDA-enabled jaxlib is not
installed" means the Python you ran has no GPU support.

## WSL2: segmentation fault when JAX starts

An older Linux NVIDIA library can load alongside the Windows driver's WSL
libraries (seen with `libnvidia-ptxjitcompiler.so.580` and a 620 series
driver). Put the matching WSL driver directory first:

```bash
find /usr/lib/wsl/drivers -name libnvidia-ptxjitcompiler.so.1 -printf '%h\n'
export LD_LIBRARY_PATH="/usr/lib/wsl/drivers/<driver-directory>${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The directory changes when the Windows driver is updated.

## RANS aborts while compiling, in `triton_tiling_propagation.cc`

Seen on JAX 0.10.2 with CUDA 13 on WSL2, with the implicit solver only.
Disabling Triton's matrix-multiplication fusion avoids the crash, and the run
stays on the GPU ([OpenXLA flags](https://openxla.org/xla/flags_guidance)):

```bash
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_triton_gemm=false"
```
