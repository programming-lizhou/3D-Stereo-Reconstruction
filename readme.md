# Dataset
Source: https://vision.middlebury.edu/stereo/data/scenes2014

# Used libraries
+ Opencv 4.6.0
+ Eigen 3.4.0
+ Ceres-solver 2.1.0

# Steps:
1. dataloader
2. feature detection
3. feature sparse matching
4. pose recovery (including 8 point alg + bundle adjustment)
5. image rectification
6. dense matching -> disparity map
7. reconstruction
8. evaluation
9. write report

# Feature detectors:
+ SIFT
+ SURF
+ ORB
+ BRISK
+ FREAK
+ KAZE

# Todo:
+ several more feature detector
+ ~~calculation fundamental matrix manually~~  (already done, but not stable sometimes)
+ bundle adjustment
+ BM dense matching (not SGBM)
+ evaluation codes

# Project structure
```
.
├── CMakeLists.txt
├── dataset
│   └── middlebury
│       ├── Bicycle1-perfect
│       │   ├── calib.txt
│       │   ├── disp0-n.pgm
│       │   ├── disp0.pfm
│       │   ├── disp0-sd.pfm
│       │   ├── disp1-n.pgm
│       │   ├── disp1.pfm
│       │   ├── disp1-sd.pfm
│       │   ├── im0.png
│       │   ├── im1E.png
│       │   ├── im1L.png
│       │   └── im1.png
│       ├── Recycle-perfect
│       │   ├── calib.txt
│       │   ├── disp0-n.pgm
│       │   ├── disp0.pfm
│       │   ├── disp0-sd.pfm
│       │   ├── disp1-n.pgm
│       │   ├── disp1.pfm
│       │   ├── disp1-sd.pfm
│       │   ├── im0.png
│       │   ├── im1E.png
│       │   ├── im1L.png
│       │   └── im1.png
│       └── ......
├── libs
│   ├── ceres-solver-2.1.0
│   └── eigen-3.4.0
├── readme.md
├── src
│   ├── bundle_adjustment.cpp
│   ├── bundle_adjustment.h
│   ├── dataloader_mb.cpp
│   ├── dataloader_mb.h
│   ├── dense_matching.cpp
│   ├── dense_matching.h
│   ├── eight_point.cpp
│   ├── eight_point.h
│   ├── feature_detector.cpp
│   ├── feature_detector.h
│   ├── main.cpp
│   ├── reconstruction.cpp
│   ├── reconstruction.h
│   ├── rectify.cpp
│   ├── rectify.h
│   ├── sparse_matching.cpp
│   ├── sparse_matching.h
│   └── useless
│       ├── dataloader_dtu.cpp
│       └── dataloader_dtu.h
└── ......
```