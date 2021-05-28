#%%
import gym
from gym import wrappers
import random
import pickle
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import os.path
import numpy as np
import time
import datetime
from datetime import datetime
from importlib import reload
import ORF_module as ORFpy
import sys
import joblib
from joblib import Parallel, delayed, parallel_backend

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Manager':
            from settings import Manager
            return Manager
        return super().find_class(module, name)


class ORF_DQN(): 
    
    def __init__(self, n_state, n_action, replay_size, ORFparams):
        self.n_action = n_action
        self.a_model = {} # to build RF for each action
        self.ORFparams = ORFparams
        self.isFit = False
        self.numTrees = ORFparams["numTrees"]
        self.maxTrees = ORFparams['maxTrees']
    
    def predict(self, s):            
        preds = [self.a_model[a].predict(s) for a in range(self.n_action)]
        return preds

    def gen_epsilon_greedy_policy(self, epsilon, n_action):        
        def policy_function(state):
            q_values =[0.0] * n_action
            
            if np.random.uniform(0,1) < epsilon:
                return([random.randint(0, n_action - 1), q_values])
            else:
                if self.isFit == True:
                    q_values = self.predict(state)
                else: 
                    return([random.randint(0, n_action - 1), q_values])
            return([np.argmax(q_values), q_values])
        
        return policy_function

    def replay(self, memory, replay_size, gamma, episode):
        
        if len(memory) >= replay_size: # When the memory size exceeds the replay_size, start updating the RFs            
            if self.isFit == False:
                for a in range(self.n_action):
                    self.a_model[a] = ORFpy.ORF(self.ORFparams, numTrees=self.numTrees) # Fit initial RFs for each action 
                self.isFit = True
            
            replay_data = random.sample(memory, replay_size) # replay_data consists of [state, action, next_state, reward, is_done]
            
            for state, action, next_state, reward, is_done in replay_data:
                
                q_values = self.predict(state)
                q_values[action] = reward + gamma * np.max(self.predict(next_state)) if is_done == False else reward
                
                # Update the RF for the action taken                
                self.a_model[action].update(state, q_values[action])    
               
            if episode == 100: # expand the number of trees at episode 100 
                for a in range(self.n_action):
                    self.a_model[a].expandTrees(self.maxTrees)


def q_learning(k, env, estimator, n_episode, replay_size, gamma=1.0, epsilon=1.0, epsilon_decay=0.95):

    global myEnv, env_file

    for episode in range(n_episode):
        
        # env = gym.wrappers.Monitor(env, directory = "./vids/"+ env_file + "_" + str(k) + "_"+ str(episode) + "/", video_callable=lambda episode_id: True, force=True)
        policy = estimator.gen_epsilon_greedy_policy(epsilon, n_action)
        state = env.reset()
        is_done = False
        episode_rewards = 0
        
        while not is_done:
            action, pred = policy(state) # integer
            next_state, reward, is_done, _ = env.step(action)

            episode_rewards += reward
            modified_reward = reward

            memory.append((state, action, next_state, modified_reward, is_done))          
            
            if is_done: # if terminated
                modified_reward = -1000 # for cartpole only
                myEnv['total_rewards'][k, episode] = episode_rewards
                estimator.replay(memory, replay_size, gamma, episode)
                break
            
            estimator.replay(memory, replay_size, gamma, episode)
            state = next_state
        
        epsilon = np.max([epsilon * epsilon_decay, 0.01])


if __name__ == "__main__":
    
    envName = "CartPole"  # Blackjack or LunarLander

    ver_num = 37

    niter = 100
    n_episode = 1000
    # env_xrng = [[1,32], [1, 11], [0, 1]] # for blackjack
    env_xrng = [[-4.8, 4.8],[-1.0, 1.0],[-0.418, 0.418],[-1.0, 1.0]] # for CartPole
    # env_xrng = [[-1.1, 1.1],[-1.1, 1.1],[-1.5, 1.5],[-1.5, 1.5],[-2.1, 2.1],[-3.1, 3.1],[0,1], [0,1]] # for LunarLander
    ORFparams = {'minSamples': 256, 'minGain': 0.05, 'xrng': env_xrng, 'maxDepth': 40, 'numTrees': 100, 'maxTrees': 200, "numTests":len(env_xrng), "gamma":1/1000} 
    QLparams = {'gamma' : 1.0, 'epsilon' : 0.5, 'epsilon_decay' : 0.99}
     
    # jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    jobid = '1'
    
    backup_file_name = envName + "_"
    backup_file = envName + "_" + str(ver_num )+ "_" + jobid

    env_file = backup_file + "_env.sav"

    myEnv = dict()
    myEnv["ORFparams"] = ORFparams
    myEnv["QLparams"] = QLparams
    myEnv["total_rewards"] = np.zeros((niter, n_episode))
    myEnv["durations"] = [0] * n_episode

    env_file = backup_file + "_env.sav"
    # pickle.dump(myEnv, open(env_file, "wb"))

    eps = QLparams["epsilon"]

    memory = deque(maxlen=10000)
    replay_size = 32
    
    def runAlg(k):
        env = gym.envs.make("CartPole-v1")        # Blackjack-v0, LunarLander-v2
        n_state = env.observation_space.shape[0]  # len(env.observation_space) for Blackjack
        n_action = env.action_space.n
        dqn = ORF_DQN(n_state, n_action, replay_size, ORFparams)
        start = time.time()
        q_learning(k, env, dqn, n_episode, replay_size, gamma=QLparams['gamma'], epsilon=eps, epsilon_decay=QLparams['epsilon_decay'])
        end = time.time()
        myEnv["durations"][k] = int(end - start)
    
    Parallel(n_jobs=4, prefer='threads')(delayed(runAlg)(k) for k in range(niter))
    pickle.dump(myEnv, open("./" + envName + "/" + env_file,"wb"))
    print(myEnv['total_rewards'])
    
#%%

# %%
