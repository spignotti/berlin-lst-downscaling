# Spatial holdout evaluation

Split: validation; metric: mae_100m in K, scene-equal. Intervals: 95% percentile, paired scene bootstrap (B=10000, rng_seed=0). n = scenes with a patch on the stratum / patches.

## MAE by stratum

| model | seeds | seen | unseen | unseen_interior | unseen_edge |
| --- | --- | --- | --- | --- | --- |
| matrix/L1 | 0,1,2 | 0.4657 [0.4244, 0.5111] (n=27/848) | 0.4430 [0.3732, 0.5122] (n=12/199) | 0.3763 [0.3249, 0.4313] (n=8/47) | 0.4490 [0.3764, 0.5202] (n=12/152) |
| matrix/L4 | 0,1,2 | 0.4459 [0.4046, 0.4886] (n=27/848) | 0.4111 [0.3517, 0.4701] (n=12/199) | 0.3340 [0.2926, 0.3780] (n=8/47) | 0.4211 [0.3582, 0.4829] (n=12/152) |

## Unseen minus seen, same model (scenes with both)

| model | difference | scenes |
| --- | --- | --- |
| matrix/L1 | 0.0258 [-0.0291, 0.0788] | 12 |
| matrix/L4 | 0.0321 [-0.0053, 0.0772] | 12 |

## L4 minus L1, matrix models

| stratum | difference | scenes |
| --- | --- | --- |
| seen | -0.0199 [-0.0412, 0.0002] | 27 |
| unseen | -0.0320 [-0.0532, -0.0138] | 12 |
| unseen_interior | -0.0423 [-0.0769, -0.0101] | 8 |
| unseen_edge | -0.0279 [-0.0500, -0.0086] | 12 |
