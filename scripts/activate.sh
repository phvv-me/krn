# Sourced on every entry into the environment. Without KRN_NVRTC_LIB the NVRTC tests and CUDA launches skip: point it
# at the directory holding libnvrtc.so.13, the nvidia wheel of this environment first, then a CUDA toolkit on the host.
if [ -z "${KRN_NVRTC_LIB:-}" ]; then
  for dir in "${CONDA_PREFIX:-/nonexistent}/lib/python3.14/site-packages/nvidia/cu13/lib" /usr/local/cuda/lib64 /opt/cuda/lib64; do
    if [ -e "$dir/libnvrtc.so.13" ]; then
      export KRN_NVRTC_LIB="$dir"
      break
    fi
  done
  unset dir
fi
