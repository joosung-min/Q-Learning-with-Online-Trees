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
import ORF_py_4 as ORFpy
import sys

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
            ran = "_"
            q_values =[0.0, 0.0]
            if np.random.uniform(0,1) < epsilon:
                ran = "Random"
                return([random.randint(0, n_action - 1), ran, q_values])
            else:
                if self.isFit == True:
                    ran = "Model"
                    q_values = self.predict(state)
                else: 
                    ran = "Random_notFit"
                    return([random.randint(0, n_action - 1), ran, q_values])
            return([np.argmax(q_values), ran, q_values])
        
        return policy_function

    def replay(self, memory, replay_size, gamma, episode):
        
        if len(memory) >= replay_size: # When the memory size exceeds the replay_size, start updating the RFs            
            
            if self.isFit == False:
                for a in range(self.n_action):
                    self.a_model[a] = ORFpy.ORF(self.ORFparams, numTrees=self.numTrees) # Fit initial RFs for each action 
                self.isFit = True
            
            replay_data = random.sample(memory, replay_size) # replay_data consists of [state, action, next_state, reward, is_done]
            
            for state, action, next_state, reward, is_done in replay_data:
                
                q_values = self.predict(state) # (, n_actions)
                q_values[action] = reward + gamma * np.max(self.predict(next_state)) if is_done == False else reward
                
                # Update the RF for the action taken                
                self.a_model[action].update(state, q_values[action])    
               
            if episode == 100: # expand the number of trees at episode 100 
                for a in range(self.n_action):
                    #self.a_model[a].discard_freq = 10 # start discarding trees more often

                    self.a_model[a].expandTrees(self.maxTrees)


def q_learning(k, env, estimator, n_episode, replay_size, gamma=1.0, epsilon=1.0, epsilon_decay=0.95, current_episode = 0):

    global myEnv, env_file

    list_episodes = list(range(n_episode))[current_episode:]
    
    for episode in list_episodes:

        # env = gym.wrappers.Monitor(env, directory = "./vids/"+ env_file + "_" + str(k) + "_"+ str(episode) + "/", video_callable=lambda episode_id: True, force=True)
        
        myEnv["episode_details"][k][episode] = []
        policy = estimator.gen_epsilon_greedy_policy(epsilon, n_action)
        state = env.reset()
        is_done = False
        i = 0

        while not is_done:
            action, ran, pred = policy(state) # integer
            next_state, reward, is_done, _ = env.step(action)
            i += 1

            myEnv["episode_details"][k][episode].append((i, state, ran, action))
            myEnv["total_rewards"][k, episode] += reward
            
            modified_reward = reward

            if is_done==True: # if terminated
                modified_reward = -100

            memory.append((state, action, next_state, modified_reward, is_done))
            if is_done:
                if estimator.isFit == True:
                    myEnv["discard_rates"][k, episode] = np.mean([estimator.a_model[0].discard_rate, estimator.a_model[1].discard_rate])
                    myEnv["memory"] = memory
                    myMods[k] = estimator
                    print("episode ", episode, " reward:", np.int(myEnv["total_rewards"][k,episode]), 
                    ", epsilon:", np.round(epsilon,3), 
                    ", meanMaxDepth:", [estimator.a_model[0].meanMaxDepth(), estimator.a_model[1].meanMaxDepth()])
                break
            estimator.replay(memory, replay_size, gamma, episode)
            state = next_state
        
        if myEnv["total_rewards"][k, episode] > 200:
            myMods[episode] = estimator
       
        epsilon = np.max([epsilon * epsilon_decay, 0.01])
        myEnv["epsilon"] = epsilon
       
        pickle.dump(myEnv, open(env_file,"wb"))
        pickle.dump(myMods, open(mod_file, "wb"))


if __name__ == "__main__":
    

    envName = "CartPole"
    ver_num = 37

    niter = 1
    n_episode = 1000
    env_xrng = [[-4.8, 4.8],[-1.0, 1.0],[-0.418, 0.418],[-1.0, 1.0]] # xrng for CartPole
    ORFparams = {'minSamples': 1024, 'minGain': 0.05, 'xrng': env_xrng, 'maxDepth': 40, 'numTrees': 100, 'maxTrees': 200, "numTests":4, "gamma":1/1000} 
    QLparams = {'gamma' : 1.0, 'epsilon' : 0.5, 'epsilon_decay' : 0.99}

    # ver13: expand numTrees from 50 to 200 at episode 100
    # ver14: do not expand trees. Constantly stay at numTrees=200
    # ver15: epsilon=0.5, epsilon_decay=0.98, minGain_decay=0.98, numTest=2, expandTrees = T
    # ver16: minSample=256
    # ver17: minSample=512, minGain_decay=0.999, numTest=2, expandTrees=T, numTrees=100 --> compare to ver8 --> bad
    # ver18: minSample=512, minGain_decay=0.999, numTest=2, expandTrees=T, discard_freq=F, numTrees=100 -> compare to 17
    # ver19: minSample=128, minGain_decay=0.999, numTest=2, expandTrees=F, discard_freq=F, numTrees=100 --> like ver8


    # ver29: minSamples = 1024, expandTrees=T --> good

    # need to try episode = 1000 for both dqn and orf(18 and 21)

    # ver30: minSamples=512, numTests=2, expandTrees=T, episode=1000 (ver18 with longer episodes)
    # ver31: minSamples=512, numTests=3, expandTrees=T, episode=1000 (ver19 with longer episodes) --> good

    # looks like increasing minSamples increases the performance also.
    # repeat ver31 50 more times, commence ver32

    # ver32: minSamples = 2048, numTests=3, expandTrees=T, episode=1000 --> not good
    
    # so far, ver29 is the best. But what if we use no modified reward?
    # ver33: minSamples=512, numTests=3, expandTrees=T, episode1000, noMR
    # ver34: minSamples=512, numTests=3, expandTrees=T, episode1000, littleMR

    # ver35: expandTrees at episode = 200

    # ver36: expandTrees at epsisode = 100, maxTrees=400

    # ver37: minSamples = 1024, numTest=n_state
     
    jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    
    import_file_rec = "C"

    if os.path.isfile(import_file_rec + "_env.sav") == True:
        recovered = True
        env_file = import_file_rec + "_env.sav"
        mod_file = import_file_rec + "_mod.sav"
        myEnv = pickle.load(open(env_file, 'rb'))
        memory = myEnv["memory"]
        eps = myEnv["epsilon"]
        resume_idx = int(np.argwhere(myEnv["total_rewards"][0] == 0)[0])-1 # where to resume
        myEnv["total_rewards"][0][resume_idx:] = 0
        myMods = CustomUnpickler(open(mod_file, 'rb')).load()
        backup_file_name = envName + "_"
        backup_file = import_file_rec

    else:
        recovered = False
        today = datetime.now()
        stamp = today.strftime("%f")
        
        backup_file_name = envName + "_"
        backup_file = envName + "_" + str(ver_num )+ "_" + jobid

        env_file = backup_file + "_env.sav"
        mod_file = backup_file + "_mod.sav"

        myEnv = dict()
        myEnv["ORFparams"] = ORFparams
        myEnv["QLparams"] = QLparams
        myEnv["total_rewards"] = np.zeros((niter, n_episode))
        myEnv["episode_details"] = {}
        for i in range(niter):
            myEnv["episode_details"][i] = {}
        myEnv["discard_rates"] = np.zeros((niter, n_episode))
        myEnv["durations"] = [0] * n_episode

        env_file = backup_file + "_env.sav"
        pickle.dump(myEnv, open(env_file, "wb"))

        mod_file = backup_file + "_mod.sav"
        myMods = dict()
        pickle.dump(myMods, open(mod_file, "wb"))
        
        eps = QLparams["epsilon"]
        resume_idx = 0
        memory = deque(maxlen=10000)

    k = 0
        
    env = gym.envs.make("CartPole-v1")        
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    replay_size = 32

    dqn = ORF_DQN(n_state, n_action, replay_size, ORFparams) if recovered == False else myMods[0]

    start = time.time()

    q_learning(k, env, dqn, n_episode, replay_size, gamma=QLparams['gamma'], epsilon=eps, epsilon_decay=QLparams['epsilon_decay'], current_episode=resume_idx)

    end = time.time()

    myEnv["durations"][k] = int(end - start)

    myMods[k] = dqn
    pickle.dump(myMods, open(mod_file, "wb"))
    pickle.dump(myEnv, open(env_file,"wb"))

    meanTR = np.mean(myEnv["total_rewards"], axis = 0)
    plt.plot(meanTR)
    plt.hlines(195, xmin=0, xmax=n_episode, linestyles="dotted", colors="gray")
    plt.title("(ORF) Average total reward per episode")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    img_name = backup_file + ".png"
    plt.savefig(fname = img_name)

# %%
