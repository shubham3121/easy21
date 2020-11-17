import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

# Card color
RED, BLACK = (-1, 1)
deck_color = [RED, BLACK]

# Deck of cards - No face cards
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def draw_card(np_random):
    """Returns the card number."""
    return np_random.choice(deck)


def card_color(np_random):
    """Returns the color of the card."""
    return np_random.choice(deck_color, p=[1 / 3.0, 2 / 3.0])


def draw_hand(np_random):
    """Returns player and dealer card choices."""
    return [draw_card(np_random), card_color(np_random)]


def sum_hand(hand):  # Return current hand total
    return sum([card * color for card, color in hand])


def is_bust(hand):  # Is this hand a bust ?
    return False if 1 <= sum_hand(hand) <= 21 else True


def score(hand):  # Calculates score of the hand
    return 0 if is_bust(hand) else sum_hand(hand)


def cmp(a, b):  # Compare the values of a and b
    return int((a > b)) - int((a < b))


class Easy21Gym(gym.Env):
    """Easy21 is an environment similar to Blackjack with following:
    1.Game is played with an infinite deck of cards.
    2.Each draw from the deck results in a value between 1 and 10.
    3.There are no aces or picture (face) cards in this game.
    """

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self._seed()
        self._reset()
        self.nA = 2

    def _reset(self):
        self.dealer = [[draw_card(self.np_random), BLACK]]
        self.player = [[draw_card(self.np_random), BLACK]]

        return self._get_obs()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return sum_hand(self.player), self.dealer[0]

    def step(self, action):
        assert self.action_space.contains(action)

        if action:  # hit: draw another card from the deck
            self.player.append(draw_hand(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_hand(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))

        return self._get_obs(), reward, done


class Easy21Numpy(object):
    """Easy21 environment using numpy only.
    Easy21 is an environment similar to Blackjack with following:
    1.Game is played with an infinite deck of cards.
    2.Each draw from the deck results in a value between 1 and 10.
    3.There are no aces or picture (face) cards in this game.
    """

    def __init__(self):
        self.actions = np.arange(2)
        self._seed()
        self.na = 2

    def _seed(self, seed=None):
        self.np_random = np.random.RandomState(seed=seed)

    @staticmethod
    def _draw_hand(np_random):
        return draw_card(np_random) * card_color(np_random)

    @staticmethod
    def _is_bust(hand_value):
        return False if 1 <= hand_value <= 21 else True

    def reset(self):
        """Returns the fully observed states for player and dealer respectively."""
        observed_state = draw_card(self.np_random) * BLACK
        return [observed_state, observed_state]

    def step(self, state, action):
        """Based on the current state and action, returns:
        1. next state of the player and dealer
        2. reward earned for the current action
        3. done state to indicate if the game is terminated.

        Args:
            state (tuple): player and dealer sums of the hand
            action (int): action (stick or hit) taken by the player.

        Returns:
            tuple: next_state, reward and done (if game is terminated or not)
        """

        player, dealer = state

        if action:  # hit: draw another card from the deck
            player += self._draw_hand(self.np_random)
            if self._is_bust(player):
                done = True
                reward = -1
            else:
                reward = 0
                done = False
        else:  # stick: play out the dealers hand, and score
            done = True
            while 0 < dealer <= 17:
                dealer += self._draw_hand(self.np_random)
            reward = cmp(player, dealer)

        next_state = [player, dealer]
        return next_state, reward, done
