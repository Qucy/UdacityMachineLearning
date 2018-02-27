# This is a unit test case for Maze.py
import numpy as np
import pandas as pd
import unittest
from Maze import Maze

class MazeTest(unittest.TestCase):

    # init
    def setUp(self):
        self.dataFrame = pd.read_csv('testworlds/world01.csv', header=None)
        self.maze_matrix = self.dataFrame.values
        self.testClass = Maze(self.maze_matrix)

    # clean
    def tearDown(self):
        pass

    
    def test_get_start_pos(self):
        start_point = (9, 4)
        self.assertEqual(start_point, self.testClass.get_start_pos(), "start point should be (9, 4)")

    def test_get_goal_pos(self):
        end_point = (0, 0)
        self.assertEqual(end_point, self.testClass.get_goal_pos(), "start point should be (0, 0)")


    def test_move(self):
        start_point = (9, 4)
        move_up = 0
        move_down = 1
        move_left = 2
        move_right = 3

        # move down newpos should be (8, 4) and reward should be -1
        newpos, reward = self.testClass.move(start_point, move_up)
        self.assertEqual((8, 4), newpos, "newpos should be (8, 4)")
        self.assertEqual(-1, reward, "reward should be -1")

        # move down newpos should be (9, 4) and reward should be -1
        newpos, reward = self.testClass.move(start_point, move_down)
        self.assertEqual(start_point, newpos, "newpos should be (9, 4)")
        self.assertEqual(-1, reward, "reward should be -1")

        # move down newpos should be (9, 3) and reward should be -1
        newpos, reward = self.testClass.move(start_point, move_left)
        self.assertEqual((9, 3), newpos, "newpos should be (9, 3)")
        self.assertEqual(-1, reward, "reward should be -1")

        # move down newpos should be (9, 5) and reward should be -1
        newpos, reward = self.testClass.move(start_point, move_right)
        self.assertEqual((9, 5), newpos, "newpos should be (9, 5)")
        self.assertEqual(-1, reward, "reward should be -1")

        # move left at (5, 0) should be (5, 0) and reward should be -1
        newpos, reward = self.testClass.move((5, 0), move_left)
        self.assertEqual((5, 0), newpos, "newpos should be (5, 0)")
        self.assertEqual(-1, reward, "reward should be -1")

        # move right at (5, 0) should be (5, 1) and reward should be -100
        newpos, reward = self.testClass.move((5, 0), move_right)
        self.assertEqual((5, 1), newpos, "newpos should be (5, 1)")
        self.assertEqual(-100, reward, "reward should be -100")

        # move right at (3, 1) should be (3, 1) and reward should be -1
        newpos, reward = self.testClass.move((3, 1), move_right)
        self.assertEqual((3, 1), newpos, "newpos should be (3, 1)")
        self.assertEqual(-1, reward, "reward should be -1")

        # move right at (0, 9) should be (0, 9) and reward should be -1
        newpos, reward = self.testClass.move((0, 9), move_right)
        self.assertEqual((0, 9), newpos, "newpos should be (0, 9)")
        self.assertEqual(-1, reward, "reward should be -1")

        # move right at (0, 9) should be (0, 9) and reward should be -1
        newpos, reward = self.testClass.move((0, 9), move_up)
        self.assertEqual((0, 9), newpos, "newpos should be (0, 9)")
        self.assertEqual(-1, reward, "reward should be -1")

        # move up at (1, 0) should be (0, 0) and reward should be 1
        newpos, reward = self.testClass.move((1, 0), move_up)
        self.assertEqual((0, 0), newpos, "newpos should be (0, 0)")
        self.assertEqual(1, reward, "reward should be 1")

if __name__ =='__main__':  
    unittest.main()