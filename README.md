# refinecbf_ros
Builds a wrapper to interface with refineCBF in ROS1. Tested in simulation with `crazyflie_clean`, for quadcopters

## Dependencies
This repository has been tested solely with ROS Noetic.
- It builds a wrapper around `crazyflie_clean`: https://github.com/UCSD-SASLab/crazyflie_clean which itself is a wrapper around `crazyflie_ros`.
- `hj_reachability`: https://github.com/StanfordASL/hj_reachability (git clone and then use `pip install -e .` for local installation. This should be using the same python version as your ROS
- `cbf_opt`: https://github.com/stonkens/cbf_opt (same as above)
- `refineCBF`: https://github.com/UCSD-SASLab/refinecbf_ros (same as above)