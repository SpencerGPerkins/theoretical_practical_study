import numpy as np
import matplotlib.pyplot as plt

class KarmedBandit:
    
    def __init__(self, k=10, eps=0.1, iterations=100):
        # Initialize k-arms, epsilon value, number of iterations
        self.k = k
        self.eps = eps
        self.iters = iterations
        
        # Initialize reward, Q table, counts per action, rewards per iter
        self.true_reward = np.random.normal(0,1, self.k)
        self.Q = np.zeros(self.k)
        self.action_counts = np.zeros(self.k)
        self.rewards = []
        
    def take_action(self):
        " Select and action using epsilon-greedy policy"
        
        if np.random.rand() < self.eps:
            return np.random.choice(self.k)
        else:
            return np.argmax(self.Q)
    
    def update_estimation(self, action, reward):
        """
        Update action-value estimate Q(action) after action selection
        """
        self.action_counts[action] += 1
        n = self.action_counts[action]
        
        self.Q[action] += (1/n) * (reward - self.Q[action])
    
    def run(self):
        """Simulate the k-armed bandit"""
        for i in range(self.iters):
            action = self.take_action()
            reward = np.random.normal(self.true_reward[action], 1)
            self.update_estimation(action, reward)
            self.rewards.append(reward)
            
    def plot_results(self, epsilon):
        """
        Plot the cumulative rewards and add a legend for epsilon.
        """
        cumulative_rewards = np.cumsum(self.rewards) / (np.arange(1, len(self.rewards) + 1))
        plt.plot(cumulative_rewards, label=f"ε = {epsilon}")
        plt.xlabel("Iterations")
        plt.ylabel("Average Reward")
        plt.title("Average Reward Over Time")
        plt.legend()  # Add a legend to the plot

k = 10
epsilon = 0.1
iterations = 1000

plt.figure(figsize=(10, 6))  # Create a figure to hold all the plots

# Parameters
for d in range(5):
    bandit = KarmedBandit(k, epsilon, iterations)
    bandit.run()
    img = bandit.plot_results(epsilon)
    epsilon *= 2
    print(f"Iter {d} Done")
    
plt.show()            
                     
            