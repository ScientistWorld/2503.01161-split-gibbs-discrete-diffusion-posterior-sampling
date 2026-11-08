# Research Direction

Improve discrete posterior sampling with a discrete diffusion prior. Within the fixed budget of 50,000 returned samples per instance, a better method returns samples whose joint distribution is closer to the exact posterior. This is measured on hidden observations, for both the coordinatewise tasks and the coupled L1-norm task.

Acceptable directions include better categorical proposals, annealing or tempering schedules, mixing improvements for coupled posteriors, better use of the diffusion prior's noised marginals, and more computation per sample where it buys accuracy.

Not acceptable: computing the exact posterior analytically and sampling it, hard-coding observations or posterior probabilities, reading evaluator code or targets, returning a different number of samples, quota or stratified tricks that match histograms instead of drawing samples, or likelihood gradients. See `scoring/CONSTRAINTS.md`.
