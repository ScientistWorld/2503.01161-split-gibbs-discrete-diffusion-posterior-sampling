# Targets

- `mnist_xor`: improve full-image PSNR and fixed-classifier consistency while maintaining low logical measurement error.
- `mnist_and_sparse`: improve these same quantities under sparse AND measurements.

Every target is present in both split score files as `psnr`, `accuracy`, and `measurement_error`. Published tables are preserved under underscore-prefixed keys in `reference.json`, because the deformed 128-image gym is not on their scale. Non-underscore reference results come from the final serialized run, not from fitting scores to the paper.
