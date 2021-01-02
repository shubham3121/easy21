import numpy as np
from collections import defaultdict
from envs.easy21 import Easy21Numpy
from utils import plot_value_function, calc_value_function


class MonteCarloControl(object):
    """Monte-Carlo control to Easy21.
    Using a time-varying scalar step-size of alpha = 1/N(s_t, a_t).
    And greedy exploration strategy with epsilon_t = N_o/(N_o + N(s_t)) where,
    - N_o = 100 is a constant.
    - N(s) = number of times the state s has been visited
    - N(s, a) = number of times the action `a` has been selected from state s.

    Plots the optimal value function V*(s) = max_a(Q*(s,a)), similar to the figure in Sutton and Barto's
    Blackjack example.
    """
    def __init__(self, N_o=100):
        self.env = Easy21Numpy()
        self.Q_val = defaultdict(lambda: np.zeros(self.env.na))
        self.n_S = defaultdict(int)
        self.n_o = N_o

    def generate_episode(self, policy=None):
        """
        Generates an episode of the Easy21 environment.
        Args:
             policy (dict): state to action map.
        Returns:
            episode (list): List of state, action and reward
        """
        episode = []
        state = self.env.reset()

        for _ in range(self.n_o):
            action = policy[state] if policy else np.random.choice(self.env.actions)
            next_state, reward, done = self.env.step(state, action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        return episode

    def mc_policy_evaluation(self, episode):
        """
        Monte Carlo Policy Evaluation. For the given episode, updates the action-value
        function.

        Args:
            episode (list): List of state, action, reward.

        """
        for state, action, reward in episode:
            self.n_S[state] += 1
            self.Q_val[state][action] += (reward - self.Q_val[state][action]) / self.n_S[state]

    def epsilon_greedy_exploration(self, state):
        """Returns the action for given state based on
        greedy exploration.
        """
        epsilon = self.n_o / (self.n_o + self.n_S[state])

        if np.random.rand() <= epsilon:
            action = np.random.choice(self.env.actions)
        else:
            action = np.argmax(self.Q_val[state])

        return action

    def mc_policy_improvement(self):
        """
        Monte-Carlo Improvement. Evaluates and updates the policy based on the current Q
        values using epsilon-greedy exploration method.

        Returns:
            dict: the new improved policy (mapping of states to actions).
        """
        policy = defaultdict(int)

        for state in self.Q_val:
            action = self.epsilon_greedy_exploration(state)
            policy[state] = action

        return policy

    def run_one_step(self, policy=None):
        """
        Evaluate and improve the policy for one step.

        Args:
            policy (dict): current policy used by the environment.
        Returns:
            dict: new improved policy.
        """
        episode = self.generate_episode(policy=policy)
        self.mc_policy_evaluation(episode)
        policy = self.mc_policy_improvement()
        return policy

    def run(self, iterations=10000):
        """
        Evaluates the action-value function and improves the policy
        for given number of iterations.

        Args:
            iterations: Number of iterations allowed.
        """
        best_policy = None
        counter = 0

        while counter < iterations:
            new_policy = self.run_one_step(policy=best_policy)
            best_policy = new_policy
            counter += 1
            if counter % int(iterations * 0.01) == 0:
                print("Progress: {:2.1%}".format(counter/iterations), end="\r")

    def plot(self):
        """
        Plots the value functions corresponding to the player
        and dealer hand values.
        """
        vs_prime = calc_value_function(self.Q_val)
        plot_value_function(vs_prime)
