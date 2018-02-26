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
            verbose = False):

        #TODO
        self.data = data
        self.reward_walk = reward_walk
        self.reward_obstacle = reward_obstacle
        self.reward_quicksand = reward_quicksand
        self.reward_goal = reward_goal
        self.random_walk_rate = random_walk_rate
        self.verbose = verbose

    #return the start position of the robot
    def get_start_pos(self):
        start_point = (np.where(self.data==2)[0][0], np.where(self.data==2)[1][0])
        return start_point

    #return the goal position of the robot
    def get_goal_pos(self):
        end_point = (np.where(self.data==3)[0][0], np.where(self.data==3)[1][0])
        return end_point

    # move the robot and report new position and reward
    # Note that robot cannot step into obstacles nor step out of the map
    # Note that robot may ignore the given action and choose a random action
    def move(self, oldpos, a):
        new_column = oldpos[0]
        new_row = oldpos[1]
        reward = 0
        # [up:a==0] [down:a==1]  [left:a==2] [right:a==3]
        # go up
        if(a == 0):
            new_column = new_column - 1
        # go down
        elif(a == 1):
            new_column = new_column + 1
        # go left
        elif(a == 2):
            new_row = new_row - 1
        # go right
        elif(a == 3):
            new_row = new_row + 1
        else:
            print("wrong a:", a)

        # boundary check, if exceed reset column and row
        if (new_column < 0 or new_column > 9 or new_row < 0 or new_row > 9):
            new_column = oldpos[0]
            new_row = oldpos[1]
            reward = -1
        # value check
        else:
            value = self.data[new_column][new_row]
            # walk into a trap
            if (value == 5):
                reward = -100
            # walk into a wall and reset column and row
            elif (value == 1):
                reward = -1
                new_column = oldpos[0]
                new_row = oldpos[1]
            # walk into an empty place or start point
            elif (value == 0 or value == 2):
                reward = -1
            # walk into destination
            elif (value == 3):
                reward = 1
            else:
                print("wrong value:", value)


        newpos = (new_column, new_row)
        # return the new, legal location and reward
        return newpos, reward

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
