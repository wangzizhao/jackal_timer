#!/bin/bash

import os
import roslaunch
import rospy
import rosnode
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String, Float32

RADIUS = 0.075

uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)

# whenever listener node receives a message, update duration
curr_duration = 0.0
trial_running = True
def callback(data):
    global curr_duration
    global trial_running

    if data.data == -1.0:
        trial_running = False
    else:
        curr_duration = float(data.data)
        #print("%f" % curr_duration)

# create node to listen to duration topic
rospy.init_node('duration_listener', anonymous=True)
rospy.Subscriber('duration', Float32, callback)

# input: row and column from the occupancy grid
# output: their corresponding position in the gazebo world
def path_coord_to_gazebo_coord(x, y):
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5

    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift

    return (gazebo_x, gazebo_y)


for num in range(0, len(os.listdir('../data/world_files'))):
    fout = open('../time_results.txt', 'a')
    
    curr_duration = 0.0
    trial_running = True

    # start and end points are currently sent as coordinates in jackal's c-space
    # with a cylinder radius of 0.075
    path = np.load('../data/path_files/path_%d.npy' % num)
    path_start = path[0]
    path_end = path[len(path)-1]

    start_x, start_y = path_coord_to_gazebo_coord(path_start[0], path_start[1])
    goal_x, goal_y = path_coord_to_gazebo_coord(path_end[0], path_end[1])

    # end point is currently provided in c-space, so we need to add in more distance
    # for it to be in the obstacle space
    # TODO - remove once start & end points are in obstacle space
    goal_y += 2 * RADIUS * 2
    start_y -= 1

    if start_x > -0.5:
        start_x = -0.5
    
    if start_x < -3.9:
        start_x = -3.9

    world_name = 'world_%d.world' % num

    args_list = ['../launch/time_trial.launch', 'world_name:=$(find jackal_timer)/data/world_files/' + world_name, 'gui:=false', 'start_x:=' + str(start_x), 'start_y:=' + str(start_y), 
        'goal_x:=' + str(goal_x), 'goal_y:=' + str(goal_y), 'config:=front_laser', 'map_file:=$(find jackal_timer)/data/map_files/yaml_' + str(num) + '.yaml']
    lifelong_args = args_list[1:]
    launch_files = [(roslaunch.rlutil.resolve_launch_arguments(args_list)[0], lifelong_args)]

    # launch the launch file
    parent = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
    parent.start()

    while trial_running and curr_duration < 50.0:
        pass

    fout.write('%d\n%f\n' % (num, curr_duration))
    fout.close()

    parent.shutdown()
    print("Finished %d" % num)
