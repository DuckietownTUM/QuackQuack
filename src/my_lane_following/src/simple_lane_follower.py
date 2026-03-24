#!/usr/bin/env python3

import os
import cv2
import numpy as np
import rospy

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped


class SimpleLaneFollower:

    def __init__(self):

        self.vehicle_name = os.environ.get("VEHICLE_NAME", "duck")

        self.prev_error = 0
        self.last_center = None

        # safer controller values
        self.kp = 2.0
        self.kd = 0.5
        self.v_bar = 0.08

        camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        cmd_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"

        self.sub = rospy.Subscriber(
            camera_topic,
            CompressedImage,
            self.callback,
            queue_size=1,
            buff_size=2**24
        )

        self.pub = rospy.Publisher(cmd_topic, Twist2DStamped, queue_size=1)

        rospy.loginfo("Lane follower started")

    def get_centroid(self, mask):

        M = cv2.moments(mask)

        if M["m00"] < 1500:
            return None

        cx = int(M["m10"] / M["m00"])
        return cx

    def publish_cmd(self, v, omega):

        cmd = Twist2DStamped()

        cmd.header.stamp = rospy.Time.now()
        cmd.v = v
        cmd.omega = omega

        self.pub.publish(cmd)

    def callback(self, msg):

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        h, w, _ = frame.shape

        roi = frame[int(h*0.55):h, :]

        roi = cv2.GaussianBlur(roi, (5,5), 0)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # better HSV thresholds
        yellow_low = np.array([10,60,60])
        yellow_high = np.array([45,255,255])

        white_low = np.array([0,0,140])
        white_high = np.array([180,90,255])

        yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)
        white_mask = cv2.inRange(hsv, white_low, white_high)

        kernel = np.ones((5,5),np.uint8)

        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        cy = self.get_centroid(yellow_mask)
        cw = self.get_centroid(white_mask)

        lane_center = None

        if cy is not None and cw is not None:
            lane_center = int((cy + cw)/2)

        elif cy is not None:
            lane_center = min(cy + 120, w)

        elif cw is not None:
            lane_center = max(cw - 120, 0)

        if lane_center is None:
            self.publish_cmd(0,0)
            return

        if self.last_center is None:
            self.last_center = lane_center

        lane_center = int(0.7*self.last_center + 0.3*lane_center)

        self.last_center = lane_center

        image_center = w//2

        error = (image_center - lane_center) / (w/2)

        d_error = error - self.prev_error

        self.prev_error = error

        raw_omega = self.kp*error + self.kd*d_error

        omega = 0.6*raw_omega + 0.4*self.prev_error

        v = max(0.05, self.v_bar*(1-abs(error)))

        self.publish_cmd(v, omega)


if __name__ == "__main__":

    rospy.init_node("simple_lane_follower")

    node = SimpleLaneFollower()

    rospy.spin()
