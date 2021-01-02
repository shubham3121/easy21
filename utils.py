import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_value_function(Q_val):
    """Calculate the value function.

    Args:
        Q_val (dict): Q val of the problem.
    Returns:
        pd.DataFrame: Vs_prime as a dataframe.
    """
    vs_prime = []
    for state in Q_val:
        player, dealer = state
        value = np.max(Q_val[state])
        vs_prime.append([int(player), int(dealer), float(value)])

    vs_prime = pd.DataFrame(data=vs_prime, columns=["player", "dealer", "value"])

    return vs_prime


def plot_value_function(vs_prime):
    """
    Plots the value function, similar to the plot in Barton and Sutton.
    Args:
        vs_prime (pd.DataFrame): value function as a DataFrame
    Reference:
        https://github.com/timbmg/easy21-rl/blob/master/utils.py
    """
    # Make the plotz
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(
        vs_prime['dealer'], vs_prime['player'], vs_prime['value'],
        cmap=plt.cm.viridis, linewidth=0.2
    )
    plt.show()

    # to Add a color bar which maps values to colors.
    surf = ax.plot_trisurf(
        vs_prime['dealer'], vs_prime['player'], vs_prime['value'],
        cmap=plt.cm.viridis, linewidth=0.2
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # Rotate it
    ax.view_init(30, 45)
    plt.show()

    # Other palette
    ax.plot_trisurf(
        vs_prime['dealer'], vs_prime['player'], vs_prime['value'],
        cmap=plt.cm.jet, linewidth=0.01
    )
    plt.show()
