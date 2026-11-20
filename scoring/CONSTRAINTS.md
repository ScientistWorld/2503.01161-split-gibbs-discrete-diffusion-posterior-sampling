# Constraints

Return exactly 128 reconstructions of 1,024 binary integer pixels for each experiment. The evaluator checks shape, dtype and category range before casting or computing metrics. It never reads method-written metrics, flags, targets, model choices, thresholds or artifact registries.

Both experiments require PSNR ≥6 dB, classifier consistency ≥20%, and logical measurement error ≤10%. Failure of any experiment or constraint assigns PSNR=0, accuracy=0 and measurement_error=1 in every experiment. A failed method therefore cannot outrank a successful method through another metric. Missing entries are also failures.

Measurements, source partitions, test size, thresholds and the frozen evaluation classifier are evaluator-owned. The method may implement any reconstruction procedure and may use public resources. It may not read hidden evaluation code or answers, change evaluation resources, or submit another method's saved outputs. Hidden inference is denied file-content reads beneath `eval/test/`, except its own run directory; directory enumeration remains available. Answers are constructed after it returns.

The paper's gradient-free claim is verified from the reference procedure, which uses inference-only model calls and discrete likelihood evaluations. A self-declared gradient-use flag is not measurable evidence and earns no score. Future attempts' compliance with scientific eligibility and hidden-data rules is checked by the separate shortcut grader; this evaluator does not claim to prove arbitrary code uses no gradients.

The scheduler's job allocation is the only execution deadline. Evaluation does not score latency or stop a method based on wall-clock load. Fixed inference counts and seeds make the recorded numbers independent of machine contention.
