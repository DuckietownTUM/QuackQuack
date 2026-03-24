#!/bin/bash
source /environment.sh
source /code/catkin_ws/devel/setup.bash
echo "Starting lane follower node..."
rosrun my_lane_following simple_lane_follower.py
