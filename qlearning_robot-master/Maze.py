"""
Template for implementing Maze
"""

import numpy as np
import random as rand

class Maze(object):

    def __init__(self,
            data,
            reward_walk = -1,
            reward_obstacle = -1,
            reward_quicksand = -100,
            reward_goal = 1,
            random_walk_rate = 0.2,
            verbose = False,
            not_disable_random = True):

        self.data = data
        self.reward_walk = reward_walk
        self.reward_obstacle = reward_obstacle
        self.reward_quicksand = reward_quicksand
        self.reward_goal = reward_goal
        self.random_walk_rate = random_walk_rate
        self.verbose = verbose
        self.move_up = 0
        self.move_down = 1
        self.move_left = 2
        self.move_right = 3
        self.min_boundary = 0
        self.max_boundary = 9
        self.empty_point = 0 # value for empty point
        self.obstacle = 1 # value for obstacle
        self.start_point = 2 # value for start point
        self.end_point = 3 # value for end point
        self.quick_sand = 5 # value for a quick_sand
        self.not_disable_random = not_disable_random # disable random move or not, for unit test only

    #return the start position of the robot
    def get_start_pos(self):
        start_point = (np.where(self.data==self.start_point)[0][0], np.where(self.data==self.start_point)[1][0])
        return start_point

    #return the goal position of the robot
    def get_goal_pos(self):
        end_point = (np.where(self.data==self.end_point)[0][0], np.where(self.data==self.end_point)[1][0])
        return end_point

    # move the robot and report new position and reward
    # Note that robot cannot step into obstacles nor step out of the map
    # Note that robot may ignore the given action and choose a random action
    def move(self, oldpos, a):
        new_row = oldpos[0]
        new_column = oldpos[1]
        reward = 0

        if (self.is_out_of_boundary(new_row, new_column)):
            raise ValueError('Current row or cloumn is out of boundary', new_row, new_column)
        
        # For UT only
        if (self.not_disable_random):
            a = self.randomly_pick_action(a)

        if(a == self.move_up):
            new_row -= 1
        elif(a == self.move_down):
            new_row += 1
        elif(a == self.move_left):
            new_column -= 1
        elif(a == self.move_right):
            new_column += 1
        else:
            raise ValueError('Current action is invalid', a)

        # boundary check after move, if exceed reset column and row
        if (self.is_out_of_boundary(new_row, new_column)):
            new_row = oldpos[0]
            new_column = oldpos[1]
            reward = self.reward_obstacle
        else:
            value = self.data[new_row][new_column]
            reward = self.retrieve_reward(value)
            # reset position if target is obstacle
            if (value == self.obstacle):
                new_row = oldpos[0]
                new_column = oldpos[1]

        newpos = (new_row, new_column)
        # return the new, legal location and reward
        return newpos, reward

    # check if out of array boundary
    def is_out_of_boundary(self, row, column):
        return column < self.min_boundary or column > self.max_boundary or row < self.min_boundary or row > self.max_boundary

    # randomly choose action
    def randomly_pick_action(self, a):
        action_array = [self.move_up, self.move_down, self.move_left, self.move_right]
        action_array.remove(a)
        p = rand.random()
        random_rate_for_per_action = self.random_walk_rate / 4
        rate_for_orignal_action = 1 - self.random_walk_rate + random_rate_for_per_action
        if (p >= 0 and p < rate_for_orignal_action):
            return a
        elif (p >= rate_for_orignal_action and p < rate_for_orignal_action + random_rate_for_per_action):
            return action_array[0]
        elif (p >= rate_for_orignal_action + random_rate_for_per_action and p < rate_for_orignal_action + random_rate_for_per_action * 2):
            return action_array[1]
        else:
            return action_array[2]


    # retrieve reward according to current point value
    def retrieve_reward(self, value):
        if (value == self.quick_sand):
            return self.reward_quicksand
        elif (value == self.obstacle):
            return self.reward_obstacle
        elif (value == self.empty_point or value == self.start_point):
            return self.reward_walk
        elif (value == self.end_point):
            return self.reward_goal
        else:
            raise ValueError('Current value is invalid', value)

    # print out the map
    def print_map(self):
        data = self.data
        print("--------------------")
        for row in range(0, data.shape[0]):
            for col in range(0, data.shape[1]):
                if data[row,col] == 0: # Empty space
                    print(" ",end="")
                if data[row,col] == 1: # Obstacle
                    print("X",end="")
                if data[row,col] == 2: # Start
                    print("S",end="")
                if data[row,col] == 3: # Goal
                    print("G",end="")
                if data[row,col] == 5: # Quick sand
                    print("~",end="")
            print()
        print("--------------------")

    # print the map and the trail of robot
    def print_trail(self, trail):
        data = self.data
        trail = data.copy()
        for pos in trail:

            #check if position is valid
            if not (    0 <= pos[0] < data.shape[0]
                    and 0 <= pos[1] < data.shape[1]):
                print("Warning: Invalid position in trail, out of the world")
                return

            if data[pos] == 1:  # Obstacle
                print("Warning: Invalid position in trail, step on obstacle")
                return

            #mark the trail
            if data[pos] == 0:  # mark enter empty space
                trail[pos] = "."
            if data[pos] == 5:  # make enter quicksand
                trail[pos] = "@"

        print("--------------------")
        for row in range(0, trail.shape[0]):
            for col in range(0, trail.shape[1]):
                if trail[row, col] == 0:  # Empty space
                    trail[row, col] = " "
                if trail[row, col] == 1:  # Obstacle
                    trail[row, col] = "X"
                if trail[row, col] == 2:  # Start
                    trail[row, col] = "S"
                if trail[row, col] == 3:  # Goal
                    trail[row, col] = "G"
                if trail[row, col] == 5:  # Quick sand
                    trail[row, col] = "~"

                print(trail[row, col], end="")
            print()
        print("--------------------")
