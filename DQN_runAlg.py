## Codes for Deep Q-learning follows instructions on 'PyTorch 1.x Reinforcement Learning Cookbook' by Y. Liu (2019)

import gym
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import os.path
import copy
import torchvision.transforms as T
from PIL import Image
import time
# from utils import mean, argmin

class DQN_exp(): 
    # def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
    def __init__(self, n_state, n_action, n_hidden=256, lr=0.001):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden), 
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action)
            )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, s, y):
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            result = self.model(torch.Tensor(s))
            return result

    def replay(self, memory, replay_size, gamma):
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size) # draw random batch
            states = []
            td_targets = []
            for state, action, next_state, reward, is_done in replay_data:
                states.append(state) # states = later converted to predicted values Q(\phi_{j}, a_{j};\theta)
                q_values = self.predict(state).tolist() # approx q-value from NN
                
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item() # q-value td update
                td_targets.append(q_values)
            self.update(states, td_targets) # gradient descent on (Q(\phi_{j}, a_{j};\theta), y_{j}) to update the weights(\theta) in the NN
    
    def gen_epsilon_greedy_policy(self, epsilon, n_action):
        def policy_function(state):
            if random.random() < epsilon:
                return random.randint(0, n_action - 1)
            else:
                q_values = self.predict(state)
                return torch.argmax(q_values).item()
        return policy_function

def q_learning(env, estimator, n_episode, replay_size, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):
    global wins, draws, losses
    for episode in tqdm(range(n_episode)):
        policy = estimator.gen_epsilon_greedy_policy(epsilon, n_action)
        state = env.reset()
        is_done = False
        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward
            memory.append((state, action, next_state, reward, is_done))
            if reward == 1:
                wins += 1
            elif reward == 0:
                draws += 1
            else:
                losses += 1
            cumulative_rewards[episode] = wins*1 + draws*0 + losses*-1
            
            if is_done:
                
                break    
                
            estimator.replay(memory, replay_size, gamma)
            state = next_state
        epsilon = max(epsilon * epsilon_decay, 0.01)


# jobid = os.getenv("SLURM_ARRAY_TASK_ID")
jobid = 1
ver = 1
envName = 'Blackjack'

backup_file = "DQN_" + envName + "_" + str(jobid)
backup_check = os.path.isfile(backup_file)

env = gym.envs.make("Blackjack-v0") # CartPole-v1, LunarLander-v2
n_state = len(env.observation_space)
# n_state = env.observation_space.shape[0] for CartPole, LunarLander
n_action = env.action_space.n
n_hidden = 64
lr = 0.005

n_episode = 1000
replay_size = 32

n_iter = 100

final_score = [0] * n_iter
c_score = {i: [] for i in range(n_iter)}
t_score = {i: [] for i in range(n_iter)}

QLparams = {'gamma' : 1.0, 'epsilon' : 0.05, 'epsilon_decay' : 1}

start = time.time()

for i in range(n_iter):
    memory = deque(maxlen=100)
    dqn = DQN_exp(n_state, n_action, n_hidden, lr)
    wins = 0
    losses = 0
    draws = 0
    total_reward_episode = [0] * n_episode
    cumulative_rewards = [0] * n_episode
    q_learning(env, dqn, n_episode, replay_size, gamma=QLparams['gamma'], epsilon=QLparams['epsilon'], epsilon_decay=QLparams['epsilon_decay']) # runs the alg
    final_score[i] = cumulative_rewards[len(cumulative_rewards)-1]
    c_score[i] = cumulative_rewards
    t_score[i] = total_reward_episode

end = time.time()
duration = int(end - start)

myEnv = dict()
# myEnv["t_r_e"] = total_reward_episode
myEnv["total_reward"] = t_score
myEnv["cumul_reward"] = c_score
# myEnv["episode_details"] = ep
# myEnv["ORFparams"] = ORFparams
myEnv["QLparams"] = QLparams
myEnv["duration"] = duration
myEnv["c_score"] = c_score
myEnv["t_score"] = t_score

with open("./" + envName + "/" + backup_file, "wb") as file:
    pickle.dump(myEnv, file)

import pandas as pd

with open(backup_file, "rb") as file:
    myEnv = pickle.load(file)

total_reward_episode = myEnv["total_reward"]
cumulative_reward = myEnv["cumul_reward"]
# ep = myEnv["episode_details"]

c_frame = pd.DataFrame(c_score)
c_frame["mean"] = c_frame.mean(axis=1)

t_frame = pd.DataFrame(t_score)
t_frame["mean"] = t_frame.mean(axis=1)
n_episode = len(t_frame["mean"])
avg_num = 100
per_h = [0] * (int(n_episode/avg_num)+1)
j = list(range(0, int(n_episode)+1, avg_num))

for i in range(len(j)-1):
    per_h[i+1] = sum(t_frame["mean"][j[i]: j[i+1]])

# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.plot(c_frame["mean"])
# plt.xlim(0,n_episode)
# plt.xlabel("episode")
# plt.ylabel("Average cumulative reward")
# plt.title("(DQN) Average Cumulative Reward") # averaged over 10 times

# plt.subplot(1,2,2)
plt.plot(per_h)
plt.xlim(1, len(per_h)-1)
plt.xlabel("100 episodes")
plt.ylabel("Average total reward")
plt.title("(DQN) Average Total Reward per " + str(avg_num) + " episodes")
# plt.savefig("(DQN)Blackjack_v1.pdf")
plt.show()


# %%
