#!/usr/bin/env python

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Path, Odometry
import rospy
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
import time

class Robot_config():
    def __init__(self):
        self.X = 0 # inertia frame
        self.Y = 0
        self.PSI = 0
        self.global_path = []
        self.gx = 0 # body frame
        self.gy = 0
        self.gp = 0
        self.los = 1
        # self.los = 5
        self.data = []
        self.num_scan_update = 0
    
    def get_robot_status(self, msg):
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w
        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y
        self.PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
        #print(self.X, self.Y, self.PSI)
    
    def get_global_path(self, msg):
        #print(msg.poses)
        # self.global_path = []
        gp = []
        for pose in msg.poses:
            gp.append([pose.pose.position.x, pose.pose.position.y])
            # self.global_path.append([pose.pose.position.x, pose.pose.position.y])
        #print(len(self.global_path))
        gp = np.array(gp)
        x = gp[:,0]
        # xhat = x
        try:
            xhat = scipy.signal.savgol_filter(x, 19, 3)
        except:
            xhat = x
        y = gp[:,1]
        # yhat = y
        try: 
            yhat = scipy.signal.savgol_filter(y, 19, 3)
        except: 
            yhat = y
        # plt.figure()
        # plt.plot(xhat, yhat, 'k', linewidth=1)
        # plt.axis('equal')
        # plt.savefig("/home/xuesu/gp_plt.png")
        # plt.close()
        gphat = np.column_stack((xhat, yhat))
        gphat.tolist()
        global_path = transform_lg(gphat, self.X, self.Y, self.PSI)
        if np.linalg.norm(global_path[:, 0]) < 10000:
            self.global_path = global_path
        # print(self.global_path)

    def get_scan(self, msg):
        scan = np.array(msg.ranges)
        if len(self.global_path) > 0:
            self.data.append([scan, self.global_path])
            self.num_scan_update += 1

def transform_lg(gp, X, Y, PSI):
    R_r2i = np.matrix([[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]])
    R_i2r = np.linalg.inv(R_r2i)
    #print(R_r2i)
    pi = np.concatenate([gp, np.ones_like(gp[:, :1])], axis=-1)
    pr = np.matmul(R_i2r, pi.T)
    return np.asarray(pr[:2, :])
    
        
if __name__ == '__main__':
    robot_config = Robot_config()
    rospy.init_node('gloabl_path', anonymous=True)
    sub_robot = rospy.Subscriber("/odometry/filtered", Odometry, robot_config.get_robot_status)
    sub_gp = rospy.Subscriber("/move_base/TrajectoryPlannerROS/global_plan", Path, robot_config.get_global_path)
    sub_scan = rospy.Subscriber("/front/scan", LaserScan, robot_config.get_scan, queue_size=1)
    lg = Pose()
    prev_num = 0
    prev_time = time.time()
    idx = 2
    goals = [[0.118966197537, -3.03216926272],
             [5.56420928299, -6.29426977547],
             [3.88057160218, -4.46378903601],
             [3.61462066307, 1.93856261749],
             [0.661892349103, 2.28965694741],
             [0.184812219794, -2.81018322449],
             [6.80693107387, -6.15469253073]]
    while not rospy.is_shutdown():
        print "[INFO]: robot pose", robot_config.X, robot_config.Y
        if len(robot_config.global_path) > 0 and np.linalg.norm(robot_config.global_path[:, 0]) > 0.2:
            print "[INFO]: bad global_path"
            break
        if (robot_config.X - goals[idx][0]) ** 2 + (robot_config.Y - goals[idx][1]) ** 2 > 0.1 ** 2:
            prev_time = time.time() 
        if (robot_config.X - 0.00206165854291) ** 2 + (robot_config.Y + 6.04767815482e-07) ** 2 < 0.1 ** 2:
            robot_config.global_path = []
            robot_config.data = []
            print "[INFO]: clean data"
        if time.time() - prev_time > 2:
            with open("/home/users/zizhaowang/jackal_ws/src/context_classifier/bag_files/3.pkl", "wb") as f:
                pickle.dump(robot_config.data, f)
            break
