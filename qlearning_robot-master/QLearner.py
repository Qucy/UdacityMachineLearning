"""
Template for implementing QLearner
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self,
        num_states=100,
        num_actions = 4,
        alpha = 0.2,
        gamma = 0.9,
        rar = 0.5,
        radr = 0.99,
        verbose = False):

        #TODO
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.Qtable = np.zeros((self.s, self.a))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = self.chooseAction(s)
        if self.verbose: print("s =", s,"a =",action)
        return action

    # I don't tink query is a good name for this method since we need to update something
    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: reward
        @returns: The selected action
        """
        # choose action
        action = self.chooseAction(s_prime)
        if self.verbose: print("s =", s_prime,"a =",action,"r =",r)
        # update Q table
        self.Qtable[self.s][action] = r + self.alhpa * self.Qtable[s_prime].argmax()
        # update alpha, rar and state
        self.alpha *= self.gamma
        self.rar *= self.radr
        self.s = s_prime

        return action

    def chooseAction(self, s):
        p = rand.random()
        if (self.rar > p):
            return rand.randint(0, self.num_actions-1)
        else:
            return self.Qtable[s].argmax()
