#%%
import gym
import torch
import random
#%%

env = gym.envs.make("PongDeterministic-v4")

state_shape = env.observation_space.shape
n_action = env.action_space.n
print(env.unwrapped.get_action_meanings())
# %%
ACTIONS = [0,2,3]
n_action = 3

env.reset()
obs = env.step(0)
is_done = False
while not is_done:
    action = ACTIONS[random.randint(0, n_action - 1)]
    obs, reward, is_done, _ = env.step(action)

import torchvision.transforms as T
from PIL import Image

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

state = get_state(obs)
print(state.shape)
# %%

from collections import deque
import copy
from torch.autograd import Variable

class DQN():
    def __init__(self, n_state, n_action, n_hidden, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[0], n_hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[1], n_action)
        )
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr
        )

    def replay(self, memory, replay_size, gamma):
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []
            for state, action, next_state, reward, is_done in replay_data:
                states.append(state.tolist())
                q_values = self.predict(state).tolist()
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.target_predict(next_state).detach()
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                td_targets.append(q_values)
            self.update(states, td_targets)

    def gen_epsilon_greedy_policy(self, epsilon, n_action):
        def policy_function(state):
            if random.random() < epsilon:
                return random.randint(0, n_action - 1)
            else:
                q_values = self.predict(state)
                return torch.argmax(q_values).item()
        return policy_function

def q_learning(env, estimator, n_episode, replay_size, target_update=1.0, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):

    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
        policy = gen_epsilon_greedy_policy(
            estimator, epsilon, n_action
        )
        obs = env.reset()
        state = get_state(obs).view(image_size * image_size).tolist()
        is_done = False
        while not is_done:
            action = policy(state)
            next_obs, reward, is_done, _ = env.step(ACTIONS[action])
            total_reward_episode[episode] += reward
            next_state = get_state(obs).view(image_size * image_size)
            memory.append((state, action, next_state, reward, is_done))

            if is_done:
                break
            estimator.replay(memory, replay_size, gamma)
            state = next_state
        print('Episode: {}, total_reward: {}, epsilon: {}'.format(episode, total_reward_episode[episode], epsilon))
        epsilon = np.max([epsilon * epsilon_decay, 0.01])

n_state = image_size * image_size
n_hidden = [200, 50]

n_episode = 1000
lr = 0.003
replay_size = 32
target_update = 10

dqn = DQN(n_state, n_action, n_hidden, lr)

memory = deque(maxlen = 100000)

total_reward_episode = [0] * n_episode
q_learning(env, dqn, n_episode, replay_size, target_update, gamma=0.9, epsilon=1)
#%%
next_state = get_state(obs).view(image_size * image_size)[0]
print(next_state)
# %%
aa = get_state(obs).view(image_size * image_size).tolist()

len(aa)
# %%
84*84
# %%
