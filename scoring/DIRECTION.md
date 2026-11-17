# Direction and weighting

Higher PSNR and fixed-classifier consistency are better; lower logical measurement error is better. A scientific improvement reconstructs unseen images more accurately while fitting the observations. Returning more images does not improve a score: it violates the output contract.

The sum uses experiment weights 0.4 (XOR) and 0.6 (sparse AND), metric coefficient magnitudes 1.0, 0.7 and 0.3, directed differences from the reference, and a shared per-metric population standard deviation across attempts. Only coefficient magnitudes matter. The standard deviation removes units; a zero-variance metric contributes zero. `eval/ranking.py` is the audit implementation.

Raw scientific quantities, not agreement with published values, are ranked. All-invalid or missing submissions receive the finite worst value in every experiment and metric before ranking. No auxiliary success flag or scalar verdict outside experiment `results` affects ranking.
