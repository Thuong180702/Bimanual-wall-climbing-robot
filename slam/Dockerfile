# FAST-LIO2 (hku-mars/FAST_LIO, pinned) on ROS Noetic, for the simulated Mid-360s.
#   docker build -t climber-slam slam/
#   python -m rl.slam_test ...          # starts the container itself (rl/slam_test.py, Slam)
FROM ros:noetic-perception
RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3-dev python3-numpy && rm -rf /var/lib/apt/lists/*
WORKDIR /catkin_ws/src
COPY livox_ros_driver livox_ros_driver
RUN git clone https://github.com/hku-mars/FAST_LIO.git && cd FAST_LIO \
    && git checkout 7cc4175de6f8ba2edf34bab02a42195b141027e9 \
    && git submodule update --init --recursive
WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && catkin_make -DCMAKE_BUILD_TYPE=Release -j4
# the bridge, its start script and the config are mounted at /sim (slam/), so editing
# them needs no rebuild
CMD ["bash", "/sim/start.sh"]
