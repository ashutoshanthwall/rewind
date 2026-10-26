# Step 6.1 — vehicle stop/start rules tuned on tuning (Kitware RGB labels)

Pre-registered in PROGRESS.md ("Phase 6 — PRE-REGISTRATION"). Tolerance 2.0 s. 384 settings; full grid: results/reports/events_6.1_grid.csv.

Edge extension: min_stopped_s at low edge -> added 0.4; min_moving_s at high edge -> added 3.0; smooth_s at high edge -> added 1.6.

## Top 10 settings (mean of the two types' pooled F1)

| unit | stop_speed | min_stopped_s | min_moving_s | smooth_s | stops P/R | starts P/R | mean F1 |
|---|---|---|---|---|---|---|---|
| box_h_per_s | 0.2 | 1.0 | 2.0 | 1.6 | 0.574 / 0.444 | 0.517 / 0.437 | 0.487 |
| px_per_s | 10.0 | 1.0 | 2.0 | 1.6 | 0.517 / 0.467 | 0.482 / 0.451 | 0.479 |
| box_h_per_s | 0.2 | 1.0 | 3.0 | 1.6 | 0.616 / 0.402 | 0.596 / 0.389 | 0.478 |
| box_h_per_s | 0.1 | 0.4 | 3.0 | 1.6 | 0.580 / 0.413 | 0.539 / 0.414 | 0.476 |
| box_h_per_s | 0.2 | 0.4 | 2.0 | 1.6 | 0.453 / 0.521 | 0.421 / 0.506 | 0.472 |
| box_h_per_s | 0.1 | 0.6 | 3.0 | 1.6 | 0.615 / 0.396 | 0.554 / 0.394 | 0.471 |
| box_h_per_s | 0.2 | 0.4 | 3.0 | 1.6 | 0.490 / 0.479 | 0.471 / 0.446 | 0.471 |
| box_h_per_s | 0.2 | 0.6 | 2.0 | 1.6 | 0.480 / 0.484 | 0.446 / 0.474 | 0.471 |
| box_h_per_s | 0.2 | 0.6 | 3.0 | 1.6 | 0.529 / 0.447 | 0.503 / 0.417 | 0.470 |
| px_per_s | 10.0 | 0.6 | 2.0 | 1.6 | 0.467 / 0.490 | 0.447 / 0.477 | 0.470 |

## Selected setting

Setting: `RuleParams(speed_unit='box_h_per_s', stop_speed=0.2, min_stopped_s=1.0, min_moving_s=2.0, smooth_s=1.6)`

**Per camera (Kitware RGB labels)**

| camera | type | detected | labelled | matched | P | R | F1 |
|---|---|---|---|---|---|---|---|
| G330 | vehicle_starts | 0 | 0 | 0 | – | – | 0.000 |
| G330 | vehicle_stops | 0 | 0 | 0 | – | – | 0.000 |
| G336 | vehicle_starts | 108 | 104 | 46 | 0.426 | 0.442 | 0.434 |
| G336 | vehicle_stops | 90 | 102 | 44 | 0.489 | 0.431 | 0.458 |
| G340 | vehicle_starts | 42 | 52 | 20 | 0.476 | 0.385 | 0.426 |
| G340 | vehicle_stops | 38 | 53 | 22 | 0.579 | 0.415 | 0.484 |
| G424 | vehicle_starts | 84 | 103 | 44 | 0.524 | 0.427 | 0.471 |
| G424 | vehicle_stops | 87 | 104 | 48 | 0.552 | 0.462 | 0.503 |
| G436 | vehicle_starts | 62 | 91 | 43 | 0.694 | 0.473 | 0.562 |
| G436 | vehicle_stops | 57 | 92 | 42 | 0.737 | 0.457 | 0.564 |
| **pooled** | vehicle_stops | 272 | 351 | 156 | **0.574** | **0.444** | 0.501 |
| **pooled** | vehicle_starts | 296 | 350 | 153 | **0.517** | **0.437** | 0.474 |

Footage scored (h): G330 0.83, G336 0.83, G340 0.75, G424 0.83, G436 0.67

**Time-shift control: every detected span moved +30 s (circular within its clip); chance level for these dense labels**

| camera | type | detected | labelled | matched | P | R | F1 |
|---|---|---|---|---|---|---|---|
| G330 | vehicle_starts | 0 | 0 | 0 | – | – | 0.000 |
| G330 | vehicle_stops | 0 | 0 | 0 | – | – | 0.000 |
| G336 | vehicle_starts | 108 | 104 | 22 | 0.204 | 0.212 | 0.208 |
| G336 | vehicle_stops | 90 | 102 | 12 | 0.133 | 0.118 | 0.125 |
| G340 | vehicle_starts | 42 | 52 | 1 | 0.024 | 0.019 | 0.021 |
| G340 | vehicle_stops | 38 | 53 | 2 | 0.053 | 0.038 | 0.044 |
| G424 | vehicle_starts | 84 | 103 | 19 | 0.226 | 0.184 | 0.203 |
| G424 | vehicle_stops | 87 | 104 | 23 | 0.264 | 0.221 | 0.241 |
| G436 | vehicle_starts | 62 | 91 | 11 | 0.177 | 0.121 | 0.144 |
| G436 | vehicle_stops | 57 | 92 | 14 | 0.246 | 0.152 | 0.188 |
| **pooled** | vehicle_stops | 272 | 351 | 51 | **0.188** | **0.145** | 0.164 |
| **pooled** | vehicle_starts | 296 | 350 | 53 | **0.179** | **0.151** | 0.164 |

**IR (G474) vs contrib labels** (incomplete labels: recall only, information only)

| type | detected | labelled | matched | R |
|---|---|---|---|---|
| vehicle_stops | 1 | 5 | 1 | 0.200 |
| vehicle_starts | 2 | 3 | 1 | 0.333 |

## Gate 6.2 part 1 (per type: pooled tuning P >= 0.5 and R >= 0.5)

- vehicle_stops: **FAIL**
- vehicle_starts: **FAIL**

Caveat: measured on the split the setting was chosen on (optimistic); dev held-out check: events_6.1_dev.md.
