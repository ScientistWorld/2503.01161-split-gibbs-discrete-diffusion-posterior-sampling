# Research Direction

Improve discrete posterior sampling for inverse problems in the same problem family as the paper. Acceptable directions include better categorical proposal mechanisms, annealing or tempering strategies, mixing improvements, and other sampling methodology that preserves the discrete posterior-sampling setup.

Do not turn the task into supervised learning from exact posterior samples, hardcode posterior probabilities, or use the evaluator's exact posterior computation inside the method. Improvements should come from better sampling methodology, not from changing the problem or leaking evaluation targets.
