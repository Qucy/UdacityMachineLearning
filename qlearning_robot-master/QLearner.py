"""
Template for implementing QLearner
"""

import random as rand
import numpy as np

class QLearner(object):

    def __init__(self,
        num_states=100,
        num_actions = 4,
        alpha = 0.2,
        gamma = 0.9,
        rar = 0.5,
        radr = 0.99,
        verbose = False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.s = 0
        self.a = 0
        self.Q = np.zeros((self.num_states, self.num_actions))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = self.choose_action(s)
        self.a = action
        if self.verbose: print("s =", s,"a =",action)
        return action


    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: reward
        @returns: The selected action
        """
        # choose action
        action = self.choose_action(s_prime)
        if self.verbose: print("s =", s_prime,"a =",action,"r =",r)
        # update Q table
        self.Q[self.s][self.a] = (1 - self.alpha) * self.Q[self.s][self.a] +  self.alpha * (r + self.alpha * self.Q[self.s][self.Q[s_prime].argmax()])
        # self.Q[self.s, self.a] = (1 - self.alpha) * (self.Q[self.s, self.a]) + self.alpha * (r + self.gamma * self.Q[s_prime,np.argmax(self.Q[s_prime,:])])
        # update rar and state
        self.rar *= self.radr
        self.s = s_prime
        self.a = action

        return action


    def choose_action(self, s):
        if (self.rar > rand.random()):
            return rand.randint(0, self.num_actions-1)
        else:
            return self.Q[s].argmax()
