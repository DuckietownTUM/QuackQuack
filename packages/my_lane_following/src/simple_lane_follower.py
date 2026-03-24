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
        self.prev_error = 0.0
        self.last_center = None

        self.parking_mode = False
        self.parked = False

        self.kp = 5.0
        self.kd = 2.5
        self.v_bar = 0.12

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

        rospy.loginfo(f"Simple lane follower started for {self.vehicle_name}")
        rospy.loginfo(f"Subscribing to: {camera_topic}")
        rospy.loginfo(f"Publishing to:  {cmd_topic}")

    def get_centroid_x(self, mask):
        M = cv2.moments(mask)
        if M["m00"] < 1000:
            return None
        return int(M["m10"] / M["m00"]) 

    def publish_cmd(self, v, omega):
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = v
        cmd.omega = omega
        self.pub.publish(cmd)

    def parallel_park(self):

        rospy.loginfo("Starting parallel parking")

        # move forward slightly
        self.publish_cmd(0.10, 0.0)
        rospy.sleep(1.5)

        # reverse while turning right
        self.publish_cmd(-0.10, -4.0)
        rospy.sleep(2.2)

        # reverse while turning left
        self.publish_cmd(-0.10, 4.0)
        rospy.sleep(2.2)

        # straighten
        self.publish_cmd(0.05, 0.0)
        rospy.sleep(1.0)

        # stop
        self.publish_cmd(0.0, 0.0)

        rospy.loginfo("Parking complete")


    def stop_robot(self):
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = 0.0
        cmd.omega = 0.0
        self.pub.publish(cmd)

    def turn_left(self):

        rospy.logininfo("Turning LEFT")
   
        self.publish_cmd(0.1, 4.0)
        rospy.sleep(2.5)

    def turn_right(self):
        rospy.logininfo("Turning RIGHT")

        self.publish_cmd(0.1, -4.0)
        rospy.sleep(2.5)

    def callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        h, w, _ = frame.shape

        roi = frame[int(h * 0.65):h, :]
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
   
        blue_low = np.array([100,120,80])
        blue_high = np.array([130,255,255 ])

        blue_mask = cv2.inRange(hsv, blue_low, blue_high)

        blue_pixels = np.sum(blue_mask)

        yellow_low = np.array([15, 80, 80])
        yellow_high = np.array([40, 255, 255])

        white_low = np.array([0, 0, 170])
        white_high = np.array([180, 70, 255])

        yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)
        white_mask = cv2.inRange(hsv, white_low, white_high)

        stop_roi = roi[int(roi.shape[0]*0.8):roi.shape[0], :]
        stop_hsv = cv2.cvtColor(stop_roi, cv2.COLOR_BGR2HSV)

        red_low1 = np.array([0,120,120])
        red_high1 = np.array([10,255,255])
   
        red_low2 = np.array([170,120,120])
        red_high2 = np.array([180,255,255])

        mask1 = cv2.inRange(stop_hsv, red_low1, red_high1)
        mask2 = cv2.inRange(stop_hsv, red_low2, red_high2)

        red_mask = mask1 + mask2

        if np.sum(red_mask) > 8000:
            rospy.loginfo("STOP LINE DETECTED")
            self.publish_cmd(0,0)
            rospy.sleep(2)
           
            return

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150)

        if np.sum(edges) > 35000:
            rospy.loginfo("Obstacle detected")
            self.publish_cmd(0.05,3.0)
            return

        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        cy = self.get_centroid_x(yellow_mask)
        cw = self.get_centroid_x(white_mask)

        lane_center = None

        if cy is not None and cw is not None:
            lane_center = int((cy + cw) / 2.0)
        elif cy is not None:
            lane_center = min(cy + 120, w - 1)
        elif cw is not None:
            lane_center = max(cw - 120, 0)

        if lane_center is None:
            rospy.loginfo("Lane lost — searching")
            self.publish_cmd(0.05, 2.5)
            return

        if lane_center is None:
            self.publish_cmd(0.0, 0.0)
            return

        if self.last_center is None:
            self.last_center = lane_center

        lane_center = int(0.7 * self.last_center + 0.3 * lane_center)
        self.last_center = lane_center

        lane_center = int(0.7*self.last_center + 0.3*lane_center)
        self.last_center = lane_center


       # Parking detection
        if blue_pixels > 7000 and not self.parked:

            rospy.loginfo("Parking spot detected")

            self.publish_cmd(0,0)
            rospy.sleep(1)

            self.parallel_park()
            return

        image_center = w // 2
        error = float(lane_center - image_center) / float(w // 2)
        d_error = error - self.prev_error
        self.prev_error = error

        omega = self.kp * error + self.kd * d_error
        omega = max(min(omega, 8.0), -8.0)
        v = self.v_bar * (1 - 1.2*abs(error))
        v = max(0.05, v)

        self.publish_cmd(v, omega)

       # cv2.imshow("camera", frame)
       # cv2.imshow("yellow", yellow_mask)
       # cv2.imshow("white", white_mask)
       # cv2.imshow("red stop", red_mask)
       # cv2.imshow("blue parking", blue_mask)

       # cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node("simple_lane_follower")
    node = SimpleLaneFollower()
    rospy.on_shutdown(node.stop_robot)
    rospy.spin()


