# COD-VAE runtime

The architecture files in this directory are derived from the official
[`join16/COD-VAE`](https://github.com/join16/COD-VAE) implementation at commit
`7b4462c7e867f1d69b25f2f5791be73df998abb3` (accessed 2026-07-18). They
retain the original module hierarchy and parameter names so the published
ShapeNet checkpoints load without conversion.

The local `pointops.py` adds a pure-PyTorch farthest-point-sampling fallback;
install COD-VAE's pointops extension to use its CUDA kernel. The occupancy head
is instantiated for checkpoint compatibility but is bypassed by Diffusion-SDF.
