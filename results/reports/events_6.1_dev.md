# Step 6.1 — vehicle stop/start rules, held-out check on dev

Frozen setting from config (selected on tuning). Information only: not part of Gate 6.2.

Setting: `RuleParams(speed_unit='box_h_per_s', stop_speed=0.2, min_stopped_s=1.0, min_moving_s=2.0, smooth_s=1.6)`

**Per camera (Kitware RGB labels)**

| camera | type | detected | labelled | matched | P | R | F1 |
|---|---|---|---|---|---|---|---|
| G330 | vehicle_starts | 0 | 0 | 0 | – | – | 0.000 |
| G330 | vehicle_stops | 0 | 0 | 0 | – | – | 0.000 |
| G336 | vehicle_starts | 10 | 10 | 8 | 0.800 | 0.800 | 0.800 |
| G336 | vehicle_stops | 10 | 12 | 8 | 0.800 | 0.667 | 0.727 |
| G340 | vehicle_starts | 6 | 2 | 0 | 0.000 | 0.000 | 0.000 |
| G340 | vehicle_stops | 2 | 2 | 0 | 0.000 | 0.000 | 0.000 |
| G424 | vehicle_starts | 12 | 10 | 4 | 0.333 | 0.400 | 0.364 |
| G424 | vehicle_stops | 12 | 11 | 4 | 0.333 | 0.364 | 0.348 |
| G436 | vehicle_starts | 10 | 18 | 7 | 0.700 | 0.389 | 0.500 |
| G436 | vehicle_stops | 12 | 17 | 9 | 0.750 | 0.529 | 0.621 |
| **pooled** | vehicle_stops | 36 | 42 | 21 | **0.583** | **0.500** | 0.538 |
| **pooled** | vehicle_starts | 38 | 40 | 19 | **0.500** | **0.475** | 0.487 |

Footage scored (h): G330 0.58, G336 0.58, G340 0.58, G424 0.58, G436 0.58

**Time-shift control: every detected span moved +30 s (circular within its clip); chance level for these dense labels**

| camera | type | detected | labelled | matched | P | R | F1 |
|---|---|---|---|---|---|---|---|
| G330 | vehicle_starts | 0 | 0 | 0 | – | – | 0.000 |
| G330 | vehicle_stops | 0 | 0 | 0 | – | – | 0.000 |
| G336 | vehicle_starts | 10 | 10 | 0 | 0.000 | 0.000 | 0.000 |
| G336 | vehicle_stops | 10 | 12 | 2 | 0.200 | 0.167 | 0.182 |
| G340 | vehicle_starts | 6 | 2 | 0 | 0.000 | 0.000 | 0.000 |
| G340 | vehicle_stops | 2 | 2 | 0 | 0.000 | 0.000 | 0.000 |
| G424 | vehicle_starts | 12 | 10 | 1 | 0.083 | 0.100 | 0.091 |
| G424 | vehicle_stops | 12 | 11 | 1 | 0.083 | 0.091 | 0.087 |
| G436 | vehicle_starts | 10 | 18 | 0 | 0.000 | 0.000 | 0.000 |
| G436 | vehicle_stops | 12 | 17 | 1 | 0.083 | 0.059 | 0.069 |
| **pooled** | vehicle_stops | 36 | 42 | 4 | **0.111** | **0.095** | 0.103 |
| **pooled** | vehicle_starts | 38 | 40 | 1 | **0.026** | **0.025** | 0.026 |

**IR (G474) vs contrib labels** (incomplete labels: recall only, information only)

| type | detected | labelled | matched | R |
|---|---|---|---|---|
| vehicle_stops | 2 | 4 | 2 | 0.500 |
| vehicle_starts | 1 | 4 | 0 | 0.000 |

