# Binary image reconstruction

Recover a 32×32 binary handwritten digit from incomplete, noiseless logical measurements. Each measurement gives the XOR or AND of a specified pair of pixels. Many images satisfy the same measurements; recovering the underlying image requires information about handwritten digits as well as measurement consistency.

There are two scored experiments: XOR with 512 pairs, and sparse AND with 2,048 pairs. Both use real MNIST images with small affine and elastic deformations, padding to 32×32, and thresholding at 128. The deformation prevents a finite catalogue of the public source images from being an answer key. Public and held-out evaluation use the same distribution and difficulty, with separate images and generation seeds.

A problem JSON supplies 128 observations, their pixel pairs, the binary vocabulary, a sampling seed, and the location of a public pretrained image model. Return one reconstruction for each observation. Any reconstruction procedure is allowed subject to the scientific eligibility rules; the public model is available but its use is not required. The hidden evaluator supplies unseen instances when it runs your procedure.
