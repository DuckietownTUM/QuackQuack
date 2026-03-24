# Duckietown – Perceptual Robustness (QuackQuack)

This project investigates the perceptual robustness of the Duckietown lane-following pipeline under real-world disturbances such as lighting variations, shadows, and calibration errors.

The project also extends the baseline system with additional perception capabilities including object detection and stop-line handling.

---

##  Team
- Aditi Deshpande  
- Tannu Malik  

---

##  Project Overview
Autonomous lane following in Duckietown relies heavily on camera-based perception. While the system performs well under ideal conditions, its performance degrades when the environment becomes unstable.

This project analyzes how disturbances affect perception modules and how these errors influence the robot's motion behavior.

---

## Objectives
- Evaluate robustness of lane-following under real-world disturbances  
- Analyze effects of lighting, shadows, and calibration errors  
- Study how perception errors propagate to control behavior  
- Extend the system with additional perception-based features  

---

## Features

### 1. Lane Following
- Implemented using Duckietown's standard pipeline  
- Core functionality for autonomous navigation  

### 2. Perceptual Robustness Analysis
Tested system under:
- **Lighting changes** → unstable detections  
- **Shadows** → missing or noisy lane detection  
- **Calibration errors** → systematic offset and incorrect steering  

### 3. Object Detection
- Detects objects during navigation  
- Enables behavior adaptation (e.g., stopping or reacting)

### 4. Stop-Line Handling
- Detects stop lines  
- Duckiebot stops for 5 seconds, then continues  

---

##  Repository Structure

```text
.
├── launchers/
│   └── default.sh
├── packages/
│   └── my_lane_following/
│       ├── CMakeLists.txt
│       ├── package.xml
│       └── src/
├── src/
│   └── my_lane_following/
│       └── src/
│           └── simple_lane_follower.py
├── modular_lane_following.py
├── dependencies-apt.txt
├── dependencies-py3.txt
├── Dockerfile
├── BPC_REPORT.pdf
├── Perceptual_Robustness_Presentation.pptx
└── README.md
