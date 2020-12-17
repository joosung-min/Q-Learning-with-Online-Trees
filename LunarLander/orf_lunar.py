#!/usr/bin/env python
# coding: utf-8

import gym
import random
import pickle
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import mean, argmax, argmin
import time
import datetime


#import ORF_py_cython as ORF
# import ORF_py as ORF
import ORF_py_2 as ORF

class ORF_DQN: 
    
    def __init__(self, n_state, n_action, replay_size, ORFparams):
        self.n_action = n_action
        self.a_model = {} # to build RF for each action
        self.a_params = {a: ORFparams for a in range(n_action)}
        self.isFit = False
        self.maxTrees = ORFparams['maxTrees']
    
    def predict(self, s):            
        # s: (4,) array (for cartpole)
        # preds = []
        # for a in range(self.n_action):
        #     preds.append(self.a_model[a].predict(s))
        
        preds = [self.a_model[a].predict(s) for a in range(self.n_action)]
        # print(preds)
        return preds

    def gen_epsilon_greedy_policy(self, epsilon, n_action):
        
        def policy_function(state):
            # state: (4,) array
            ran = "_"
            q_values =[0.0, 0.0]
            if random.uniform(0,1) < epsilon:
                ran = "Random"
                return([random.randint(0, n_action - 1), ran, q_values])
            else:
                if self.isFit == True:
                    ran = "Model"
                    q_values = self.predict(state) # (1,2) array
                    # print(q_values)
                else: 
                    ran = "Random_notFit"
                    return([random.randint(0, n_action - 1), ran, q_values])
                    # print("passed random.randint")
            return([argmax(q_values), ran, q_values])# int
        
        return policy_function

    def replay(self, memory, replay_size, gamma, episode):

        if len(memory) == replay_size: # Build initial Forests
            
            for a in range(self.n_action):
                self.a_params[a]['xrng'] = ORF.dataRange([v[0] for v in memory if v[1] == a])
                self.a_model[a] = ORF.ORF(self.a_params[a]) # Fit initial RFs for each action            

        if len(memory) >= replay_size: # When the memory size exceeds the replay_size, start updating the RFs            
            
            replay_data = random.sample(memory, replay_size) # replay_data consists of [state, action, next_state, reward, is_done]
            for state, action, next_state, reward, is_done in replay_data:
                
                q_values = self.predict(state) # (, n_actions)
                q_values[action] = reward + gamma * max(self.predict(next_state)) if is_done == False else reward
                
                # Update the RF for the action taken
                xrng = ORF.dataRange([v[0] for v in replay_data if v[1] == action])
                self.a_model[action].update(state, q_values[action], xrng)    
            self.isFit = True
               
        if episode == 100: # expand the number of trees at episode 100            
            # expandForest(memory)
            for a in range(self.n_action):
                self.a_params[a]['xrng'] = ORF.dataRange([v[0] for v in memory if v[1] == a])
                lenFor = len(self.a_model[a].forest)
                for i in range(lenFor+1, self.maxTrees):
                    # self.a_model[a].forest[i] = ORF.ORT(self.a_params[a]) # build new empty trees
                    best_tree_idx = argmin([ORF.ORF.forest[j].OOBError for j in range(len(ORF.ORF.forest))])
                    self.a_model[a].forest[i] = ORF.ORF.forest[best_tree_idx] # duplicate the best tree

def q_learning(env, estimator, n_episode, replay_size, gamma=1.0, epsilon=0.1, epsilon_decay=0.95):
    
    for episode in tqdm(range(n_episode)):
        policy = estimator.gen_epsilon_greedy_policy(epsilon, n_action)
        state = env.reset()
        is_done = False
        i = 0
        while not is_done:
            action, ran, pred = policy(state) # integer
            next_state, reward, is_done, _ = env.step(action)
            i += 1
            # next_state: 4x1 array (for cartpole)
            # reward: integer
            # is_done: bool (True/False)

            total_reward_episode[episode] += reward
            
            # Modified rewards for mtcar depending on its location 
            # assign larger reward for being close to the right side
            
            ## Modified rewards for lunarlander?
            modified_reward = reward
            
            if -0.05 < next_state[1] < 0.05: 
                modified_reward = 500
                if -0.05 < next_state[3] < 0.05: # modified reward for y
                    modified_reward = 1000
            
            if -0.05 < next_state[0] < 0.05: # modified reward for x coord
                modified_reward = 500
                if -0.05 < next_state[2] < 0.05: # modified reward for x velocity
                    modified_reward = 1000

            if reward == 100: # modified reward for land / crash
                modified_reward = 10000
            elif reward == -100:
                modified_reward = -10000

            ep[episode].append((i, state, ran, action, reward))
            memory.append((state, action, next_state, modified_reward, is_done))
            # memory.append((state, action, next_state, reward, is_done))
            
            if is_done:
                break
            estimator.replay(memory, replay_size, gamma, episode)
            state = next_state
        epsilon = max([epsilon * epsilon_decay, 0.001])
        # print(epsilon)


# Initialization
env = gym.envs.make("LunarLander-v2")
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

memory = deque(maxlen=5000)
n_episode = 1000
replay_size = 32

ORFparams = {'minSamples': replay_size*5, 'minGain': 0.1, 'xrng': None, 'maxDepth': 50, 'numTrees': 5, 'maxTrees': 50} 
# default = memory = 10000, minSamples = replay_size*5, minGain = 0.01, maxDepth = 30
# lunar noMR 3: memory=5000, minSamples replay_size*5, minGain=0.1, maxDepth=70, epsilon_decay=0.99
# lunar MR 1: memory = 5000, minSamples = replay_size*10, minGain=0.1, maxDepth = 50

backup_file_name = "lunar_MR" + "_2"

dqn = ORF_DQN(n_state, n_action, replay_size, ORFparams) 

total_reward_episode = [0] * n_episode

ep = {i: [] for i in range(n_episode)}

QLparams = {'gamma' : 1.0, 'epsilon' : 1.0, 'epsilon_decay' : 0.99}



# Run alg

start = time.time()

q_learning(env, dqn, n_episode, replay_size, gamma=QLparams['gamma'], epsilon=QLparams['epsilon'], epsilon_decay=QLparams['epsilon_decay']) # runs the alg

end = time.time()

duration = int(end - start)

print("learning duration =", duration, " seconds")
print("mean reward = ", mean(total_reward_episode))
print("max reward = ", max(total_reward_episode))


# In[27]:


img_file = backup_file_name + ".jpg"
backup_file_p = backup_file_name + ".p"

myEnv = dict()
myEnv["ep"] = ep
myEnv["ORFparams"] = ORFparams
myEnv["QLparams"] = QLparams
myEnv["tre"] = total_reward_episode

with open(backup_file_p, "wb") as file:
    pickle.dump(myEnv, file)

plt.plot(total_reward_episode)
plt.title("(ORF) Total reward per episode")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.hlines(200, xmin=0, xmax=n_episode, linestyles="dotted", colors="gray")
plt.show()
plt.savefig(fname = img_file)



# About LunarLander
"""
Rocket trajectory optimization is a classic topic in Optimal Control.
According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points.

Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
on its first attempt. Please see the source code for details.
To see a heuristic landing, run:
python gym/envs/box2d/lunar_lander.py
To play yourself, run:
python examples/agents/keyboard_agent.py LunarLander-v2
Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

observation_space = (x coord, y coord, x velocity, y velocity, lander angle, angular velocity, right-leg grounded, left-leg grounded)
action_space = do nothing(0), fire left engine(1), fire mian engine(2), fire right engine(3)
"""




