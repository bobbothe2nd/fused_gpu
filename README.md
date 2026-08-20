# `fused_gpu`

Advanced graph-based GPU compiler for linear algebra and AI/ML/DL.

## Project Status

Some functionality/tests are broken, and I'm currently rebuilding the traversal architecture.

`softmax` or any custom `!compute_gid` node does not work forward or backward.

Basic tested operations include:

- `sub`
- `matmul`
- `add`
- `mul`

Other unary/binary operations are likely trivially correct but not extensively tested.

## Backends

Only supports WGPU/WGSL backends. CUDA and ROCm backends planned before `v1.0.0`. Other backends (e.g. CPU) are unlikely. Custom backends are fully supported.
