# Frozen model implementation

Transformer, rotary and residual operations come from the authors' SGDD repository,
commit `d3d4d60e331a98ca7a72ff06d12164635a45465d`, `models/SEDD/`:
https://github.com/chuwd19/Split-Gibbs-Discrete-Diffusion-Posterior-Sampling

The only architecture change is replacing FlashAttention with PyTorch SDPA for
portable inference; weights, rotary layout, full attention and all layers are
unchanged. Removed optional Hugging Face mixin, optional rotary extension, and
process-wide JIT settings. The published EMA checkpoint is used without training.
The sampler's prior integrator follows the authors' released analytic predictor
and final denoiser; the paper calls its few-step prior solver Euler.
