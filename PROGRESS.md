# Progress

The final image gym is complete through **core_claim_plus**. Job `cba2826c-061` finished with exit 0 after resuming unchanged raw outputs from a lost worker. No reference model, input distribution, inference setting or metric changed during recovery.

The reference implements the authors' released SGDD procedure with the original full-resolution 90M MNIST SEDD checkpoint and paper K/T/H settings. Hidden XOR achieves PSNR 17.978222 dB, class consistency 0.859375, measurement error 0.001770; sparse AND achieves 12.520452 dB, 0.632812, 0.002026. Public-slice results are comparable. Scaled SMC and SVDD-MC, plus full-budget direct MH, have lower raw reconstruction quality and fail the task gates.

The paired stronger fallback gains +0.160067 with a 95% interval [+0.031717, +0.291394], exceeding the observed independent-seed gain of +0.016194. The fixed-coupling weaker variant loses −1.670260. Every replayed shortcut scores below the reference; logical solving gains −0.670145 and exhaustive training-template search −0.522951. No reference metric is at its best bound.

`scoring/scores_train.json`, `scores_test.json` and `scores.json` contain only the canonical reference and three comparators. Active reference values match the serialized hidden run; the paper's reported values are unchanged in archival keys. The sampler registry and ordinary end-to-end launcher are restored. Bundled correctness checks and `python validate.py --compare` passed. Two validator path-name heuristics warn despite both evaluators explicitly loading raw reconstruction files.

`HARDENING.md` contains before/after scores, the REVIEW.md fixes, paired uncertainty, remaining limitations and resource scaling. The original DPS baseline and DNA/music/FFHQ studies are outside this reproduction. Continuous deformation-aware source lookup was not exhaustively tested; arbitrary-code gradient eligibility remains with the separate shortcut grader.

Earlier synthetic reproduction milestones are historical, recorded in `MILESTONES.md` and git history. The factorized and chain tasks no longer affect ranking because their exact posteriors and simple MH solutions invalidated scientific headroom.
