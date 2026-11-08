# Experiments

All experiments ask for 50,000 posterior samples per problem instance and are scored on hidden observations. `hellinger` and `total_variation` measure the first two coordinates (the paper's Table 2 measurement). `all_pairs_hellinger` measures every pair of coordinates.

## synthetic_d2

The smallest Table 2 setting. Each coordinate's posterior is clearly bimodal and far from the prior, so this tests whether a sampler can move mass to where the observation puts it.

## synthetic_d5

The middle Table 2 setting. More nuisance coordinates share one observation-to-latent problem. This tests whether accuracy holds as dimension grows.

## synthetic_d10

The largest Table 2 setting. With many coordinates, any per-coordinate bias accumulates over the joint and the all-pairs check.

## synthetic_l1_d10

Added experiment, not in the paper's table. Ten coordinates are observed only through one noisy scalar summary of all of them, with the observation in the prior's upper tail. The posterior couples every coordinate and sits far from the prior. This tests whether a sampler stays accurate when the posterior is not a product of independent pieces, which is where the paper's own synthetic settings show the most room for improvement.
