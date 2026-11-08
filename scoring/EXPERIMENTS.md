# Experiments

All experiments ask for 50,000 posterior samples per problem instance and are scored on hidden observations. `hellinger` and `total_variation` measure the first two coordinates (the paper's Table 2 measurement). `all_pairs_hellinger` measures every pair of coordinates.

## synthetic_d2

The smallest Table 2 setting. Each coordinate's posterior is clearly bimodal and far from the prior, so this tests whether a sampler can move mass to where the observation puts it.

## synthetic_d5

The middle Table 2 setting. More nuisance coordinates share one observation-to-latent problem. This tests whether accuracy holds as dimension grows.

## synthetic_d10

The largest Table 2 setting. With many coordinates, any per-coordinate bias accumulates over the joint and the all-pairs check.

## synthetic_chain_d10

Added experiment, not in the paper's table. Ten coordinates are observed only through the distances between neighbours, which mirrors the paper's pairwise measurements on images. Many configurations explain the data almost equally well: steps can go either way, and the whole chain can shift. The prior decides between them. This tests whether a sampler can find and correctly weight many separated modes when the posterior does not split into independent pieces. It carries the largest weight because it is where accurate posterior sampling is hardest and where the paper's own settings leave the most room for improvement.
