# monto carlo every visit

#%%
import numpy as np
import gym
from collections import defaultdict
import sys

#%%
env = gym.make("Blackjack-v0")  # make environemnt from gym name called blackjack-v0
print("[+] stated {}".format(env.observation_space))  # 704
print("[+] action {}".format(env.action_space.n))  # 2


# %%
# stochastic probability
# montercalo perdiction
# 18 > hit 20% none 80%
# 18 < hit 80% none 20%


def generate_episode_form_stochastic(env):
    episode = []
    state = env.reset()
    while True:
        prob = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(
            np.arange(2), p=prob
        )  # be carefull form the parameters
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


for i in range(3):
    print(generate_episode_form_stochastic(env))


# %%
# mc prediction
# first visit and the every visit : (every visit)


def mc_perdiction(env, num_episode, generate_episode, gamma=1.0):
    reutrn_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for i_episode in range(1, num_episode + 1):
        if i % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = generate_episode(env)
        states, action, reward = zip(*episode)
        discount = np.array([gamma ** i for i in range(len(reward) + 1)])

        for i, state in enumerate(states):
            # action value function
            reutrn_sum[state][action[i]] += sum(reward[i] * discount[: (-1 + i)])
            print(N[state][action[i]])
            N[state][actions[i]] += 1.0
            # updating the Q values using montercarlo every visit
            Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]
    return Q
