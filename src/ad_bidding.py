import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)    # Number of times each arm was selected
        self.values = np.zeros(n_arms)    # Estimated value for each arm

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Incremental update
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

class UCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self):
        self.total_counts += 1
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
            bonus = np.sqrt((2 * np.log(self.total_counts)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta prior parameters for each arm
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)

    def select_arm(self):
        theta_samples = np.random.beta(self.successes, self.failures)
        return np.argmax(theta_samples)

    def update(self, chosen_arm, reward):
        # Reward should be binary: 1 for success, 0 for failure
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1

def simulate_bidding(algorithm, n_rounds=500, true_ctr=None):
    """
    Simulate bidding optimization over a number of rounds.
    :param algorithm: instance of one of the MAB classes.
    :param n_rounds: number of bidding rounds.
    :param true_ctr: list of true CTRs (rewards) for each ad (arm).
    """
    n_arms = algorithm.n_arms
    if true_ctr is None:
        # For demonstration, assume fixed CTRs for each ad
        true_ctr = np.random.uniform(0.05, 0.3, n_arms)
    rewards = []
    for _ in range(n_rounds):
        chosen_arm = algorithm.select_arm()
        # Simulate click outcome based on true CTR probability
        reward = np.random.binomial(1, true_ctr[chosen_arm])
        algorithm.update(chosen_arm, reward)
        rewards.append(reward)
    total_reward = np.sum(rewards)
    print(f"Total reward after {n_rounds} rounds: {total_reward}")
    return rewards

if __name__ == "__main__":
    n_ads = 5
    print("Epsilon-Greedy Simulation:")
    eg = EpsilonGreedy(n_ads, epsilon=0.1)
    simulate_bidding(eg)

    print("\nUCB Simulation:")
    ucb = UCB(n_ads)
    simulate_bidding(ucb)

    print("\nThompson Sampling Simulation:")
    ts = ThompsonSampling(n_ads)
    simulate_bidding(ts)