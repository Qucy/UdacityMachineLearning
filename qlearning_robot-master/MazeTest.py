import numpy as np
import pandas as pd
import unittest
from Maze import Maze
from mazeqlearning import to_state

# Unit test case
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
        self.testClass = Maze(self.maze_matrix, not_disable_random = False)
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


    # test invalid cases
    def test_move_with_invalid_argu(self):
        self.assertRaises(ValueError, self.testClass.move, (9, 9), 5)


    def test_randomly_pick_action(self):
        action0_count = 0
        action1_count = 0
        action2_count = 0
        action3_count = 0

        total_times = 1000000

        for i in range(total_times):
            a = self.testClass.randomly_pick_action(0) # choose action 0 as orginal action
            if (a == 0):
                action0_count += 1
            elif (a == 1):
                action1_count += 1
            elif (a == 2):
                action2_count += 1
            else:
                action3_count += 1
        self.assertTrue(action0_count < total_times * 0.86 and action0_count > total_times * 0.84, "action 0 count should between 0.84~0.86 of the data")
        self.assertTrue(action1_count < total_times * 0.06 and action1_count > total_times * 0.04, "action 1 count should between 0.04~0.06 of the data")
        self.assertTrue(action2_count < total_times * 0.06 and action2_count > total_times * 0.04, "action 2 count should between 0.04~0.06 of the data")
        self.assertTrue(action3_count < total_times * 0.06 and action3_count > total_times * 0.04, "action 3 count should between 0.04~0.06 of the data")

        #print("\n action 0 count:", action0_count, "\n action 1 count:", action1_count, "\n action 2 count:", action2_count, "\n action 3 count", action3_count)

    def test_to_state(self):
        self.assertEqual(to_state((0, 0)), 0, "state should be 0")
        self.assertEqual(to_state((0, 1)), 1, "state should be 1")
        self.assertEqual(to_state((1, 9)), 19, "state should be 19")
        self.assertEqual(to_state((2, 0)), 20, "state should be 20")
        self.assertEqual(to_state((9, 9)), 99, "state should be 99")

if __name__ =='__main__':
    unittest.main()
