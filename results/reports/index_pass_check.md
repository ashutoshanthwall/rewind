# index_pass output check

Clips done: 300 of 300. Leftover .tmp files: 0 (only expected for the clip being written right now).

**Problems: 1 in 1 clips.**

- 2018-03-15.15-40-04.15-45-04.bus.G340: detections: boxes outside the 1920x1080 frame by more than 3% of its size

Per camera (sums over done clips):

| camera | modality | clips | seconds | activity_mean | motion_boxes | n_person | n_car | n_truck | n_bus | n_motorcycle | n_bicycle | detections |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G330 | rgb | 51 | 15351 | 0.0066 | 470416 | 724253 | 7059 | 0 | 0 | 2 | 83 | 731397 |
| G336 | rgb | 51 | 15313 | 0.0038 | 182187 | 102847 | 854343 | 79491 | 961 | 128 | 2554 | 1040324 |
| G340 | rgb | 49 | 14737 | 0.0051 | 103119 | 114188 | 797860 | 106094 | 1651 | 43 | 831 | 1020667 |
| G424 | rgb | 50 | 15023 | 0.0039 | 180837 | 137315 | 1626385 | 238828 | 9762 | 317 | 978 | 2013585 |
| G436 | rgb | 48 | 14401 | 0.0045 | 247179 | 289084 | 329720 | 34259 | 3399 | 2445 | 4289 | 663196 |
| G474 | ir | 51 | 15336 | 0.0051 | 246753 | 14855 | 23076 | 3498 | 50 | 98 | 512 | 42089 |

Clips with zero activity or zero detections (3; not an error by itself):

- 2018-03-07.16-55-00.17-00-00.school.G330: activity mean 0.00000, zero-activity seconds 298/301, motion boxes 1, detections 0
- 2018-03-07.17-25-00.17-30-00.school.G330: activity mean 0.00000, zero-activity seconds 301/301, motion boxes 0, detections 0
- 2018-03-15.15-55-00.16-00-00.school.G330: activity mean 0.00000, zero-activity seconds 301/301, motion boxes 0, detections 1312

## Reviewed by Claude (25 Oct, 23:50)

- **The 1 flagged clip is detector behaviour, not a pipeline error.** 2018-03-15.15-40-04.15-45-04.bus.G340: 50
  `car` boxes (frames 4860-5328) extend up to 479 px below the 1080 px frame. Viewed by Claude (n = 2 frames, 4920
  and 5220): a red car parked right in front of the camera is cut by the bottom edge, and YOLOX predicts the full car
  extent past the edge. In the same frames, the box around the visible part of the car fits it, so the scale-back is
  correct. Every other clip overhangs by at most 28 px. Consequence: step 4.2 must clip boxes to the frame before
  cropping. The check's 3 % tolerance is kept unchanged on purpose, so a real scaling bug would still trip it.
- **Zero-activity clips.** G330 03-07 16:55 and 17:25 show an empty gym (viewed at the mid-run check).
  G330 03-15 15:55 has 0 activity but 1,312 `person` detections, one per sample. Viewed by Claude (n = 1, frame
  3000): a bag lying at the bottom edge, detected as `person` (scores 0.30-0.53). This is a static false positive,
  the second found after the 03-11 bleacher `car`s.
