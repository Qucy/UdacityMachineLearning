"""
Train a Q Learner in a navigation problem.
"""

import pandas as pd
import numpy as np
import random as rand
import QLearner as ql
import Maze


# convert the position to a single integer state
# only work for 10 * 10 matrix : )
def to_state(pos):
    row = pos[0]
    column = pos[1]
    if (row == 0):
        return column
    else:
        return int(str(row) + str(column))


# train learner to go through maze multiple epochs
# each epoch involves one trip from start to the goal or timeout before reaching the goal
# return list of rewards of each trip
def train(maze, learner, epochs=500, timeout = 100000, verbose = False):
    rewards = np.zeros(epochs)
    goal = maze.get_goal_pos()
    for i in range(epochs):
        current_time = 0
        total_reward = 0
        robopos = maze.get_start_pos()
        action = learner.querysetstate(to_state(robopos))
        while (robopos != goal and current_time < timeout):
            newpos, reward = maze.move(robopos, action)
            if (-100 == reward):
                print("epochs:", i, "S:", robopos, "A:", action, " S':", newpos, "R:", reward)
            robopos = newpos
            action = learner.query(to_state(robopos), reward)
            total_reward += reward
            current_time += 1
        #print(i," reward:", total_reward)
        rewards[i] = total_reward

    print(rewards)
    return rewards


# run the code to train a learner on a maze
def maze_qlearning(filename):
    #initialize maze object
    data_frame = pd.read_csv(filename, header=None)
    maze = Maze.Maze(data_frame.values, verbose=False)
    #initialize learner object
    qlearner = ql.QLearner(verbose=False)
    #execute train(maze, learner)
    rewards = train(maze, qlearner, epochs=1, timeout=1000)
    #return median of all rewards
    return rewards

if __name__=="__main__":
    rand.seed(5)
    maze_qlearning('testworlds/world01.csv')
