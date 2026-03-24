ARG REPO_NAME=dt-core
ARG REPO_TAG=daffy-arm64v8

FROM duckietown/${REPO_NAME}:${REPO_TAG}

# Fix ROS GPG key expiration
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key F42ED6FBAB17C654

COPY dependencies-apt.txt /tmp/dependencies-apt.txt
RUN apt-get update && xargs -r apt-get install -y < /tmp/dependencies-apt.txt && rm -rf /var/lib/apt/lists/*

COPY dependencies-py3.txt /tmp/dependencies-py3.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/dependencies-py3.txt

COPY ./packages /code/catkin_ws/src
COPY ./launchers /launchers

RUN . /opt/ros/noetic/setup.sh && catkin build

CMD ["/bin/bash", "/launchers/default.sh"]
