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
driver). Put the matching WSL driver directory first. Discover it at shell
startup instead of hard-coding the Windows driver-store suffix in `.bashrc`:

```bash
if [[ -d /usr/lib/wsl/drivers ]]; then
    _jax_wsl_ptx=(/usr/lib/wsl/drivers/*/libnvidia-ptxjitcompiler.so.1)
    if (( ${#_jax_wsl_ptx[@]} == 1 )) && [[ -f ${_jax_wsl_ptx[0]} ]]; then
        _jax_wsl_driver_dir=${_jax_wsl_ptx[0]%/*}
        export LD_LIBRARY_PATH="${_jax_wsl_driver_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    else
        printf '%s\n' 'CUDA WSL: expected one driver directory; selection is ambiguous or missing.' >&2
    fi
    unset _jax_wsl_ptx _jax_wsl_driver_dir
fi
```

Replace the previous hard-coded export with this block, rather than appending
it. After a Windows driver update, open a new shell to rediscover the directory
before starting Python. Existing Python processes must be restarted too.
If more than one candidate exists, the block reports the ambiguity instead of
assuming the first/newest directory is the active driver. This is a workaround
for mixed WSL/Linux NVIDIA libraries; `/usr/lib/wsl/lib` alone did not resolve
the observed JAX startup crash.

## RANS aborts while compiling, in `triton_tiling_propagation.cc`

Seen on JAX 0.10.2 with CUDA 13 on WSL2, with the implicit solver only.
Disabling Triton's matrix-multiplication fusion avoids the crash, and the run
stays on the GPU ([OpenXLA flags](https://openxla.org/xla/flags_guidance)):

```bash
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_triton_gemm=false"
```
