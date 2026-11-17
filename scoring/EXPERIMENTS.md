# Scored experiments

`mnist_xor` measures recovery of real binary handwritten digits from 512 XOR measurements of random pixel pairs. The measurements determine relations but leave many pixels and component orientations ambiguous. The experiment tests whether recovered images preserve both the underlying digit and the observed logical relations.

`mnist_and_sparse` measures recovery from 2,048 AND measurements of random pixel pairs. Zeros are ambiguous and sparse positive measurements leave substantial missing information. This is the sparse regime of the paper's empirical prior-value study. It tests reconstruction quality where observations alone do not identify the digit.

Each experiment uses 128 independently deformed MNIST images, with the same instances for every method. Both report whole-image PSNR, fixed-classifier consistency with the underlying image, and the fraction of measurement bits violated. The last quantity prevents a visually plausible but observation-inconsistent image from counting as successful reconstruction. Validity and task gates are documented in `CONSTRAINTS.md`.

XOR has weight 0.4 and sparse AND 0.6. Sparse AND receives greater weight because it tests the paper's missing-information claim and provides a difficult reconstruction regime. Raw metric coefficients are PSNR 1.0, class consistency 0.7, and measurement error 0.3; contributions are normalized by variation across attempts. There are no scored runtime, declared-flag, posterior-histogram, or agreement-with-published-number metrics.
