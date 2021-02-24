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
import glob
import ORF_py_4 as ORFpy
#%%

env = gym.envs.make("PongDeterministic-v4")

# state_shape = env.observation_space.shape
# n_action = env.action_space.n
# print(env.unwrapped.get_action_meanings())

# ACTIONS = [0,2,3]
# n_action = 3

# env.reset()
# obs = env.step(0)
# is_done = False
# while not is_done:
#     action = ACTIONS[random.randint(0, n_action - 1)]
#     obs, reward, is_done, _ = env.step(action)

# import torchvision.transforms as T
# from PIL import Image

image_size = 84
transform = T.Compose([T.ToPILImage(), 
T.Grayscale(num_output_channels=1), 
T.Resize((image_size, image_size), interpolation=Image.CUBIC),
T.ToTensor(),])

def get_state(obs):
    state = obs.transpose((2,0,1))
    state = torch.from_numpy(state)
    state = transform(state)
    return state

# state = get_state(obs)
# print(state.shape)

class ORF_DQN(): 
    
    def __init__(self, n_state, n_action, replay_size, ORFparams):
        self.n_action = n_action
        self.a_model = {} # to build RF for each action
        self.a_params = {a: ORFparams for a in range(n_action)}
        self.isFit = False
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
                    self.a_model[a] = ORFpy.ORF(self.a_params[a]) # Fit initial RFs for each action 
                self.isFit = True
            
            replay_data = random.sample(memory, replay_size) # replay_data consists of [state, action, next_state, reward, is_done]
            
            for state, action, next_state, reward, is_done in replay_data:
                
                q_values = self.predict(state) # (, n_actions)
                q_values[action] = reward + gamma * np.max(self.predict(next_state)) if is_done == False else reward
                
                # Update the RF for the action taken                
                self.a_model[action].update(state, q_values[action])    
               
            if episode == 100: # expand the number of trees at episode 100 
                for a in range(self.n_action):
                    # self.a_model[a].discard_freq = 1 # start discarding trees more often
                    self.a_model[a].expandTrees(self.maxTrees)
                    # self.a_model[a].replaceTrees()


def q_learning(k, env, estimator, n_episode, replay_size, gamma=1.0, epsilon=0.1, epsilon_decay=0.95, current_episode = 0):

    global myEnv, env_file

    list_episodes = list(range(current_episode, n_episode))
    
    for episode in list_episodes:
       
        # env = gym.wrappers.Monitor(env, directory = "./vids/"+ env_file + "_" + str(k) + "_"+ str(episode) + "/", video_callable=lambda episode_id: True, force=True)
        
        myEnv["episode_details"][k][episode] = []
        policy = estimator.gen_epsilon_greedy_policy(epsilon, n_action)
        obs = env.reset()
        state = get_state(obs).view(image_size * image_size).tolist()
        
        # state = env.reset()
        is_done = False
        i = 0

        while not is_done:
            action, ran, pred = policy(state) # integer
            next_obs, reward, is_done, _ = env.step(action)
            next_state = get_state(next_obs).view(image_size * image_size)
            
            # next_state, reward, is_done, _ = env.step(action)
            i += 1

            # myEnv["episode_details"][k][episode].append((i, state, ran, action))
            # myEnv["total_rewards"][k, episode] += reward
            
            modified_reward = reward

            # if is_done==True: # if terminated
            #     modified_reward = -1000

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

#%%
if __name__ == "__main__":
    

    envName = "Pong"
    ver_num = 0

    niter = 1
    n_episode = 10
    n_state = image_size * image_size
    # env_xrng = [[-4.8, 4.8],[-1.0, 1.0],[-0.418, 0.418],[-1.0, 1.0]] # xrng for CartPole
    env_xrng = [[0, 255]] * (image_size * image_size)
    ORFparams = {'minSamples': 512, 'minGain': 0.05, 'xrng': env_xrng, 'maxDepth': 30, 'numTrees': 50, 'maxTrees': 100, "numTests":image_size*image_size, "gamma":0.1} 
    QLparams = {'gamma' : 1.0, 'epsilon' : 1.0, 'epsilon_decay' : 0.98}

    # file_path = os.getcwd() +'/*'
    # file_list = glob.glob(file_path)
    # # file_name = [file[:-8] for file in file_list if file.endswith("_env.sav")][0]
    # png_file = file_name + ".png"
    

    recovered = False
    today = datetime.now()
    stamp = today.strftime("%f")
    
    backup_file_name = envName + "_"
    backup_file = envName + "_" + str(ver_num )+ "_" + stamp 

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
        
    env = gym.envs.make("PongDeterministic-v4")        
    # n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    replay_size = 32

    dqn = ORF_DQN(n_state, n_action, replay_size, ORFparams) if recovered == False else myMods[len(myMods)-1]

    start = time.time()

    q_learning(k, env, dqn, n_episode, replay_size, gamma=QLparams['gamma'], epsilon=eps, epsilon_decay=QLparams['epsilon_decay'], current_episode=resume_idx)

    end = time.time()
    print("here!")
    myEnv["durations"][k] = int(end - start)

    # myMods[k] = dqn
    pickle.dump(myMods, open(mod_file, "wb"))
    pickle.dump(myEnv, open(env_file,"wb"))

    meanTR = np.mean(myEnv["total_rewards"], axis = 0)
    plt.plot(meanTR)
    plt.hlines(195, xmin=0, xmax=n_episode, linestyles="dotted", colors="gray")
    plt.title("(ORF) Average total reward per episode")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    img_name = backup_file + ".pdf"
    plt.savefig(fname = img_name)

# %%
