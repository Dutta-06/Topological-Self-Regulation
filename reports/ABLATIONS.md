# Growth ablations (final-epoch top-1, %)

## resnet18_cifar100  (seeds present: [42, 43, 44])

| arm | kind | n | top-1 | Δ vs same-seed Ref | params |
|---|---|---|---|---|---|
| ref | ref | 3 | 77.63 ± 0.20 | -- | 11,220,132 |
| released | online | 3 | 77.55 ± 0.17 | -0.08 ± 0.36 | 9,536,707 |
| released | c2 | 3 | 77.42 ± 0.46 | -0.20 ± 0.58 | 9,536,707 |
| fixed | online | 3 | 77.30 ± 0.17 | -0.33 ± 0.32 | 9,536,825 |
| fixed | c2 | 3 | 77.25 ± 0.25 | -0.37 ± 0.44 | 9,536,825 |
| off | online | 3 | 77.18 ± 0.26 | -0.45 ± 0.43 | 9,533,186 |
| off | c2 | 3 | 77.41 ± 0.25 | -0.22 ± 0.40 | 9,533,186 |
| random | online | 3 | 77.41 ± 0.13 | -0.22 ± 0.33 | 9,536,231 |
| cost | online | 3 | 77.24 ± 0.19 | -0.39 ± 0.38 | 9,536,823 |
| oto | online | 3 | 77.20 ± 0.26 | -0.42 ± 0.11 | 9,533,098 |

Paired same-seed differences, a − b (points); `wins` = seeds where a > b.

| a − b | kind | n | mean ± s.d. | wins | question |
|---|---|---|---|---|---|
| released − off | online | 3 | 0.37 ± 0.15 | 3/3 | does growth help, as published? |
| released − off | c2 | 3 | 0.01 ± 0.59 | 1/3 | does growth help, as published? |
| fixed − off | online | 3 | 0.12 ± 0.12 | 2/3 | does growth help, with measured-direction births? |
| fixed − off | c2 | 3 | -0.16 ± 0.27 | 1/3 | does growth help, with measured-direction births? |
| fixed − released | online | 3 | -0.25 ± 0.15 | 0/3 | does keeping the measured birth direction help? |
| fixed − released | c2 | 3 | -0.17 ± 0.34 | 1/3 | does keeping the measured birth direction help? |
| fixed − random | online | 3 | -0.11 ± 0.17 | 1/3 | does the gradient choose the growth site better than chance? |
| fixed − cost | online | 3 | 0.06 ± 0.10 | 1/3 | does the gradient beat 'grow the cheapest group'? |
| released − oto | online | 3 | 0.34 ± 0.41 | 2/3 | TSR vs OTOv2 (one run from scratch, same count) |
| off − oto | online | 3 | -0.02 ± 0.44 | 1/3 | removal-only TSR vs OTOv2 (one run from scratch, same count) |

Growth mechanics (online arms, from decision logs).

| arm | seed | births | share in top group | groups grown | births' ports | actions |
|---|---|---|---|---|---|---|
| released | 42 | 261 | 100% | 1 | {'grad': 155, 'random': 106} | {'pure_prune': 188, 'pure_grow': 131, 'exchange': 130} |
| released | 43 | 247 | 100% | 1 | {'grad': 126, 'random': 121} | {'pure_prune': 188, 'pure_grow': 105, 'none': 1, 'exchange': 142} |
| released | 44 | 251 | 100% | 1 | {'grad': 120, 'random': 131} | {'pure_prune': 190, 'pure_grow': 100, 'exchange': 151} |
| fixed | 42 | 254 | 100% | 1 | {'grad': 254} | {'pure_prune': 178, 'pure_grow': 119, 'exchange': 135} |
| fixed | 43 | 247 | 100% | 1 | {'grad': 247} | {'pure_prune': 188, 'pure_grow': 105, 'none': 1, 'exchange': 142} |
| fixed | 44 | 261 | 100% | 1 | {'grad': 261} | {'pure_prune': 187, 'pure_grow': 118, 'exchange': 143} |
| off | 42 | 0 | 0% | 0 | {} | {'pure_prune': 159, 'none': 254} |
| off | 43 | 0 | 0% | 0 | {} | {'pure_prune': 171, 'none': 244} |
| off | 44 | 0 | 0% | 0 | {} | {'pure_prune': 170, 'none': 248} |
| random | 42 | 238 | 34% | 11 | {'grad': 238} | {'pure_prune': 204, 'pure_grow': 119, 'exchange': 119, 'none': 3} |
| random | 43 | 220 | 30% | 10 | {'grad': 220} | {'pure_prune': 213, 'pure_grow': 114, 'none': 6, 'exchange': 106} |
| random | 44 | 237 | 37% | 9 | {'grad': 237} | {'pure_prune': 206, 'pure_grow': 121, 'none': 4, 'exchange': 116} |
| cost | 42 | 262 | 100% | 1 | {'grad': 262} | {'pure_prune': 186, 'pure_grow': 105, 'exchange': 157, 'none': 1} |
| cost | 43 | 247 | 100% | 1 | {'grad': 247} | {'pure_prune': 188, 'pure_grow': 105, 'none': 1, 'exchange': 142} |
| cost | 44 | 261 | 100% | 1 | {'grad': 261} | {'pure_prune': 187, 'pure_grow': 118, 'exchange': 143} |

Discovery stability across seeds: cosine similarity of log2(width/ref) vectors (final widths), raw and modulo cost-free swaps between interchangeable groups ([[1, 3]]), and the most-grown group per seed.

| arm | seeds | mean pairwise cosine | modulo swaps | most-grown group |
|---|---|---|---|---|
| released | 3 | 0.364 (min 0.038) | 0.993 (min 0.991) | differs: [3, 1, 1] |
| fixed | 3 | 0.362 (min 0.046) | 0.995 (min 0.994) | differs: [3, 1, 1] |
| off | 3 | 0.984 (min 0.982) | 0.984 (min 0.982) | differs: [5, 5, 1] |
| random | 3 | 0.975 (min 0.963) | 0.983 (min 0.979) | differs: [3, 1, 3] |
| cost | 3 | 0.996 (min 0.994) | 0.996 (min 0.994) | same: [1, 1, 1] |

## vgg16bn_cifar100  (seeds present: [42, 43, 44])

| arm | kind | n | top-1 | Δ vs same-seed Ref | params |
|---|---|---|---|---|---|
| ref | ref | 3 | 73.46 ± 0.26 | -- | 14,774,436 |
| released | online | 3 | 73.27 ± 0.58 | -0.19 ± 0.47 | 12,554,172 |
| released | c2 | 3 | 73.18 ± 0.02 | -0.28 ± 0.25 | 12,554,172 |
| fixed | online | 3 | 73.19 ± 0.72 | -0.28 ± 0.63 | 12,554,433 |
| fixed | c2 | 3 | 73.17 ± 0.04 | -0.30 ± 0.25 | 12,554,433 |
| off | online | 3 | 73.27 ± 0.39 | -0.19 ± 0.36 | 12,555,150 |
| off | c2 | 3 | 73.28 ± 0.21 | -0.19 ± 0.07 | 12,555,150 |
| random | online | 3 | 73.14 ± 0.53 | -0.32 ± 0.41 | 12,557,139 |
| cost | online | 3 | 73.19 ± 0.72 | -0.28 ± 0.63 | 12,554,433 |
| oto | online | 3 | 73.18 ± 0.37 | -0.29 ± 0.25 | 12,554,891 |

Paired same-seed differences, a − b (points); `wins` = seeds where a > b.

| a − b | kind | n | mean ± s.d. | wins | question |
|---|---|---|---|---|---|
| released − off | online | 3 | 0.00 ± 0.20 | 1/3 | does growth help, as published? |
| released − off | c2 | 3 | -0.10 ± 0.19 | 1/3 | does growth help, as published? |
| fixed − off | online | 3 | -0.09 ± 0.33 | 1/3 | does growth help, with measured-direction births? |
| fixed − off | c2 | 3 | -0.11 ± 0.19 | 1/3 | does growth help, with measured-direction births? |
| fixed − released | online | 3 | -0.09 ± 0.16 | 1/3 | does keeping the measured birth direction help? |
| fixed − released | c2 | 3 | -0.01 ± 0.02 | 0/3 | does keeping the measured birth direction help? |
| fixed − random | online | 3 | 0.05 ± 0.21 | 2/3 | does the gradient choose the growth site better than chance? |
| fixed − cost | online | 3 | 0.00 ± 0.00 | 0/3 | does the gradient beat 'grow the cheapest group'? |
| released − oto | online | 3 | 0.10 ± 0.71 | 2/3 | TSR vs OTOv2 (one run from scratch, same count) |
| off − oto | online | 3 | 0.10 ± 0.60 | 2/3 | removal-only TSR vs OTOv2 (one run from scratch, same count) |

Growth mechanics (online arms, from decision logs).

| arm | seed | births | share in top group | groups grown | births' ports | actions |
|---|---|---|---|---|---|---|
| released | 42 | 190 | 100% | 1 | {'grad': 190} | {'pure_prune': 302, 'pure_grow': 176, 'exchange': 14} |
| released | 43 | 113 | 100% | 1 | {'grad': 112, 'random': 1} | {'pure_prune': 297, 'none': 79, 'pure_grow': 103, 'exchange': 10} |
| released | 44 | 193 | 100% | 1 | {'grad': 190, 'random': 3} | {'pure_prune': 312, 'pure_grow': 176, 'exchange': 17} |
| fixed | 42 | 190 | 100% | 1 | {'grad': 190} | {'pure_prune': 302, 'pure_grow': 176, 'exchange': 14} |
| fixed | 43 | 113 | 100% | 1 | {'grad': 113} | {'pure_prune': 297, 'none': 79, 'pure_grow': 103, 'exchange': 10} |
| fixed | 44 | 193 | 100% | 1 | {'grad': 193} | {'pure_prune': 312, 'pure_grow': 176, 'exchange': 17} |
| off | 42 | 0 | 0% | 0 | {} | {'pure_prune': 302, 'none': 190} |
| off | 43 | 0 | 0% | 0 | {} | {'pure_prune': 296, 'none': 193} |
| off | 44 | 0 | 0% | 0 | {} | {'pure_prune': 311, 'none': 194} |
| random | 42 | 176 | 36% | 13 | {'grad': 176} | {'pure_prune': 316, 'pure_grow': 121, 'exchange': 55} |
| random | 43 | 100 | 47% | 10 | {'grad': 100} | {'pure_prune': 308, 'none': 81, 'pure_grow': 75, 'exchange': 25} |
| random | 44 | 177 | 44% | 12 | {'grad': 177} | {'pure_prune': 328, 'pure_grow': 135, 'exchange': 42} |
| cost | 42 | 190 | 100% | 1 | {'grad': 190} | {'pure_prune': 302, 'pure_grow': 176, 'exchange': 14} |
| cost | 43 | 113 | 100% | 1 | {'grad': 113} | {'pure_prune': 297, 'none': 79, 'pure_grow': 103, 'exchange': 10} |
| cost | 44 | 193 | 100% | 1 | {'grad': 193} | {'pure_prune': 312, 'pure_grow': 176, 'exchange': 17} |

Discovery stability across seeds: cosine similarity of log2(width/ref) vectors (final widths), raw and modulo cost-free swaps between interchangeable groups (none in this architecture), and the most-grown group per seed.

| arm | seeds | mean pairwise cosine | modulo swaps | most-grown group |
|---|---|---|---|---|
| released | 3 | 0.994 (min 0.990) | 0.994 (min 0.990) | same: [0, 0, 0] |
| fixed | 3 | 0.993 (min 0.990) | 0.993 (min 0.990) | same: [0, 0, 0] |
| off | 3 | 0.939 (min 0.911) | 0.939 (min 0.911) | same: [1, 1, 1] |
| random | 3 | 0.974 (min 0.965) | 0.974 (min 0.965) | same: [0, 0, 0] |
| cost | 3 | 0.993 (min 0.990) | 0.993 (min 0.990) | same: [0, 0, 0] |

