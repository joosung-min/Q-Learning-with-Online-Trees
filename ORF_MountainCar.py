import gym
import random
import pickle
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
# import os.path
# import copy
import numpy as np
import time
import datetime
import sys
# import multiprocessing
# from cProfile import Profile


# # On Google Colab
# # # -------------------------------------------------------------------------
# from google.colab import drive
# drive.mount('/content/drive/')
# sys.path.insert(0, '/content/drive/My\ Drive/Colab\ Notebooks/')
# # # --------------------------------------------------------------------------
# !pip install box2d-py


# In[4]:


# !pip install gym[Box_2D]


# In[5]:


# !python setup_DQN.py build_ext --inplace


# In[6]:


# !cython -a ORF_cython.pyx
# !cython -a DQN.pyx
# %cd /content/drive/My\ Drive/ShallowQlearning/
# !python setup_ORF.py build_ext --inplace
import ORF_cython as ORF


# In[9]:


# Initialization
# env = gym.envs.make("LunarLander-v2")
# env=gym.envs.make("CartPole-v1")
env = gym.envs.make("MountainCar-v0")
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
memory = deque(maxlen=100000)
n_episode = 600
replay_size = 32

ORFparams = {'minSamples': replay_size*5, 'minGain': 0.1, 'xrng': None, 'maxDepth': 50, 'numTrees': 5, 'maxTrees': 30} # numTrees -> 30 after 100 iters. 25 restarts

dqn = ORF.ORF_DQN(n_state, n_action, replay_size, ORFparams) 

total_reward_episode = np.zeros(n_episode)

ep = {i: [] for i in range(n_episode)}

QLparams = {'gamma' : 1.0, 'epsilon' : 1.0, 'epsilon_decay' : 0.99}


# In[10]:


# Run alg

start = time.time()

result = ORF.q_learning(env, dqn, n_episode, n_action, memory, replay_size, gamma=QLparams['gamma'], epsilon=QLparams['epsilon'], epsilon_decay=QLparams['epsilon_decay']) # runs the alg

end = time.time()

duration = int(end - start)

print("learning duration =", duration, " seconds")
print("mean reward = ", np.mean(total_reward_episode))
print("max reward = ", max(total_reward_episode))


# 

# In[ ]:


# backup_file_name = "ORF_LunarLander_" + time.strftime("%y%m%d") + "_1"
backup_file_name = time.strftime("%y%m%d") + "MountainCar_iter1"
img_file = backup_file_name + ".jpg"
plt.plot(result)
plt.title("(ORF) Total reward per episode")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.hlines(195, xmin=0, xmax=n_episode, linestyles="dotted", colors="gray")
plt.show()
plt.savefig(fname = img_file)


# In[ ]:


# To back-up the work
backup_file = backup_file_name + ".p"
backup_check = os.path.isfile(backup_file)

myEnv = dict()
myEnv["t_r_e"] = result
myEnv["duration"] = duration
myEnv["episode_details"] = ep
myEnv["ORFparams"] = ORFparams
myEnv["QLparams"] = QLparams
# myEnv["ORF_params"] = params

with open(backup_file, "wb") as file:
    pickle.dump(myEnv, file)


# In[ ]:


# About MountainCar
# Actions: discrete(3) 0:left, 1:no push, 2: push right
# Reward: -1 for each time step, until the goal position of 0.5 is reached.
# Starting state: Random position from -0.6 to 0.4 with no velocity
# Episode termination: ends when the car reaches 0.5 position, or if 200 iterations are reached.
# Solve requirement: get average reward of -110.0 over 100 consecutive trials.


