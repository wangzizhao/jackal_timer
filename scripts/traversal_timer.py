#!/usr/bin/env python

from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Path, Odometry
import rospy
import numpy as np
from numpy import inf
import time
import sys


class Robot():
    def __init__(self):
        self.X = 0 # inertia frame
        self.Y = 0
        self.PSI = 0

    def get_robot_status(self, msg):
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w
        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y
        self.PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
        

if __name__ == '__main__':
    goal_x = float(sys.argv[1])
    goal_y = float(sys.argv[2])

    jackal = Robot()

    time_pub = rospy.Publisher('duration', Float32, queue_size=10)
    rospy.init_node('traversal_timer', anonymous=True, disable_signals=True)

    sub_robot_status = rospy.Subscriber("/odometry/filtered", Odometry, jackal.get_robot_status)

    start_time = rospy.get_time()

    while not rospy.is_shutdown():
        duration = rospy.get_time() - start_time
        time_pub.publish(Float32(duration))
        if ((jackal.X-goal_x)**2)+((jackal.Y-goal_y)**2) < 0.1:
            rospy.signal_shutdown("Done")