#!/bin/bash
# inside the container: a ROS master, FAST-LIO's parameters, the bridge (which runs FAST-LIO)
. /catkin_ws/devel/setup.bash
roscore > /tmp/roscore.log 2>&1 &
until rostopic list > /dev/null 2>&1; do sleep 0.2; done
rosparam load /sim/sim_mid360.yaml
exec python3 /sim/bridge.py
