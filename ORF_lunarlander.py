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


# In[3]:


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
import ORF_py_cython as ORF


# In[9]:


# Initialization
env = gym.envs.make("LunarLander-v2")
# env=gym.envs.make("CartPole-v1")
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
memory = deque(maxlen=10000)
n_episode = 2000
replay_size = 32

ORFparams = {'minSamples': replay_size*5, 'minGain': 0.1, 'xrng': None, 'maxDepth': 70, 'numTrees': 5, 'maxTrees': 30} # numTrees -> 30 after 100 iters. 25 restarts

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
backup_file_name = "ORF_LunarLander_" + time.strftime("%y%m%d") + "_iter2"
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


# About Cartpole
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    
    Reward:
        Reward is 1 for every step taken, including the termination step
    
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
    
    Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """


# In[ ]:


# import pstats
# stats = pstats.Stats("profile_200913.pfl")
# stats.strip_dirs()
# stats.sort_stats('cumulative')
# stats.print_stats()


# In[ ]:




