import numpy as np


def compute_gae(rewards, values, gamma: float = 0.99, lam: float = 0.95):
    """
    Calculates the unnormalized generalized advantage estimation function.
    Check out https://arxiv.org/abs/1506.02438.

    Arguments:
    rewards : ndarray of size (N,)
     - rewards earned during a trajectory
    values : ndarray of size(N+1,)
     - calculated values of states during a trajectory
    gamma / lam: float
     - discounting constants described in paper

    Returns:
    adv : ndarray of size 
     - generalized advantage estimation
    """
    assert len(values) == len(rewards) + 1

    N = len(rewards)
    adv = np.zeros((N,))
    gae = 0

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] - values[step]
        gae = delta + gamma * lam * gae
        adv[step] = gae
    
    return adv


def compute_rtg(rewards, gamma: float = 0.99):
    """
    Calculates the unnormalized sum of rewards after a point in a trajectory.
    Here we add a discounting factor that puts less emphasis on results that
    happen further in the future.

    Arguments:
    rewards : ndarray of size (N,)
     - rewards earned during a trajectory
    gamma : float
    - discounting factor (closer to one implies more weight for future events)

    Returns:
    returns : ndarray
    - returns of action, or the discounted future rewards
    """
    N = len(rewards)
    returns = np.zeros((N,))

    for step in reversed(range(N)):
        future_rewards = gamma * (returns[step+1] if step+1 < N else 0)
        returns[step] = rewards[step] + future_rewards
    return returns
