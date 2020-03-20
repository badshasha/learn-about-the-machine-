import numpy as np


class machine:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.experiment_mean = 0
        self.representate = 0

    def work(self):
        return np.random.randn() + self.true_mean

    def update(self, reward):
        self.representate += 1
        self.experiment_mean = (
            1 - 1.0 / self.representate
        ) * self.experiment_mean + 1.0 / self.representate * reward


def experiment(array_of_machine, eps, N):
    bandits = [machine(b) for b in array_of_machine]
    for i in range(N):
        p = np.random.random()
        j = (
            np.random.choice(len(bandits))
            if p < eps
            else np.argmax([b.experiment_mean for b in bandits])
        )
        reward = bandits[j].work()
        bandits[j].update(reward)

    for i, b in enumerate(bandits):
        print(
            "machine  {} true mean is {} used {}".format(
                i, b.experiment_mean, b.representate
            )
        )


experiment([1.0, 2.0, 3.0], 0.1, 10000)

