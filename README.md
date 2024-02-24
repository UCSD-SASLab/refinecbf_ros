# refinecbf_ros
Builds a wrapper to interface with refineCBF in ROS1. Tested in simulation with `crazyflie_clean`, for quadcopters

## Dependencies
This repository has been tested solely with ROS Noetic.
- It builds a wrapper around `crazyflie_clean`: https://github.com/UCSD-SASLab/crazyflie_clean which itself is a wrapper around `crazyflie_ros`.
- `hj_reachability`: https://github.com/StanfordASL/hj_reachability (git clone and then use `pip install -e .` for local installation. This should be using the same python version as your ROS
- `cbf_opt`: https://github.com/stonkens/cbf_opt (same as above)
- `refineCBF`: [https://github.com/UCSD-SASLab/refineCBF](https://github.com/UCSD-SASLab/refineCBF) (same as above)
- `matplotlib`: For visualization purposes, a matplotlib version of 3.6.2 is reccomended. Use `pip install matplotlib==3.6.2` to install this version.

## Turtlebot
To use the Turtlebot examples in this repository, please install the necessary Turtlebot3 packages using [these instructions](https://automaticaddison.com/how-to-launch-the-turtlebot3-simulation-with-ros/).
## Crazyflie 
To use the Crazyflie examples in this repository, please clone the [crazyflie_clean package from UCSD SASLab](https://github.com/UCSD-SASLab/crazyflie_clean).
## Jackal
To use the Jackal examples in this repository, please install `jackal-simulator` for ROS Noetic

`jackal-simulator`: `sudo apt-get install ros-noetic-jackal-simulator ros-noetic-jackal-desktop ros-noetic-jackal-navigation`
