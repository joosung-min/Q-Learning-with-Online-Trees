# In[20]:


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
import datetime
from datetime import datetime
import os
import numpy as np

class DQN_exp(): 
    def __init__(self, n_state, n_action, n_hidden=128, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden), 
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action)
            )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr, )

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

def q_learning(k, env, estimator, n_episode, replay_size, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):
    
    global myEnv, env_file

    for episode in tqdm(range(n_episode)):
        policy = estimator.gen_epsilon_greedy_policy(epsilon, n_action)
        state = env.reset()
        is_done = False

        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            # total_reward_episode[episode] += reward

            myEnv["total_rewards"][k, episode] += reward
            modified_reward = reward
            
            y_coord = next_state[1]
            y_velo = next_state[3]
            left_leg = next_state[6]
            right_leg = next_state[7]
            if is_done == 1 and y_coord == 0 and y_velo == 0 and left_leg==1 and right_leg == 1: # perfect landing 
                modified_reward = 1000
            
            # memory.append((state, action, next_state, reward, is_done))
            memory.append((state, action, next_state, modified_reward, is_done))
            
            if is_done:
                pickle.dump(myEnv, open(env_file, "wb"))
                break
            
            estimator.replay(memory, replay_size, gamma)
            state = next_state
        epsilon = max(epsilon * epsilon_decay, 0.1)


# In[28]:

today = datetime.now()
stamp = today.strftime("%f")


# jobid = os.getenv("SLURM_ARRAY_TASK_ID")
jobid = '1'
backup_file = "DQN_LunarLander_" + jobid
# backup_check = os.path.isfile(backup_file)

n_hidden = 32
lr = 0.005

iter = 10
n_episode = 1000
replay_size = 32


myEnv = dict()
myEnv["total_rewards"] = np.zeros((iter, n_episode))
myEnv["durations"] = [0] * n_episode

env_file = backup_file + "_env.sav"
pickle.dump(myEnv, open(env_file, "wb"))

for k in range(iter):
    env = gym.envs.make("LunarLander-v2")
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    dqn = DQN_exp(n_state, n_action, n_hidden, lr)
    memory = deque(maxlen=10000)

    start = time.time()
    q_learning(k, env, dqn, n_episode, replay_size, gamma=0.9, epsilon=0.5, epsilon_decay=0.99)
    end = time.time()
    duration = int(end - start)

myEnv["DQN_params"] = [n_hidden, lr, replay_size]

with open(env_file, "wb") as file:
    pickle.dump(myEnv, file)

meanTR = np.mean(myEnv["total_rewards"], axis=0)

# print(meanTR)

img_file = backup_file + ".png"
print(duration)
plt.plot(meanTR)
plt.title("(DQN) Total reward per episode")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.show()
plt.savefig(img_file)


# In[ ]:




