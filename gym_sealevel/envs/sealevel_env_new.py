import gym
from gym import error, spaces, utils
from gym.utils import seeding
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, A2C, PPO, SAC
# from stable_baselines.common.cmd_util import make_vec_env
import numpy as np
import matplotlib.pyplot as plt

class SealevelEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, init_slr, years, max_cost):
        super(SealevelEnv, self).__init__()
        self.init_slr = init_slr #sealevel in current year
        self.years = years #number of years to run environment
        self.max_cost = max_cost
        self.curr_slr = self.init_slr
        self.curr_year = 0
        self.curr_cost = 0
        self.r = 0.04 #maybe add discount rate attribute needed for optimal policy

        self.max_slr = 2500 #sea level rise by 2100 with the highest greenhouse gas emissions according to climate.gov
        low = np.array([0, 0, self.init_slr]) 
        high = np.array([self.years, self.max_cost, self.max_slr]) #(year, cost, sea level rise)
        n_actions = 3 #0 for protect, 1 for retreat/ do nothing, 2 for mitigate

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)
       
    def step(self, action):
        if action == 0:
            self.curr_cost += 0.158400000
            self.curr_slr += 3.6
        if action == 1:
            self.curr_cost += 0.921600000 
            self.curr_slr += 3.6
        if action == 2:
            self.curr_cost += 0.937500000
            self.curr_slr += 1.0
        # self.curr_slr += 3.6
        self.curr_year += 1
        done = bool(self.curr_slr >= self.max_slr 
        or self.curr_year == self.years 
        or self.curr_cost >= self.max_cost)
        #reward = (1 - self.curr_slr / self.max_slr) + (1 -  self.curr_cost / self.max_cost) #take into account both cost and sea level
        reward = 1/(1/(1+self.r)**self.curr_year*self.curr_cost) #minimizing cost model
        #reward = 1/self.curr_slr #minimizing damage model
        return np.array([self.curr_year, self.curr_cost, self.curr_slr]), reward, done, {}
    
    def reset(self):
        self.curr_slr = self.init_slr
        self.curr_year = 0
        self.curr_cost = 0
        return np.array([self.curr_year, self.curr_cost, self.curr_slr])
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

env = SealevelEnv(0.1, 100, 100.0) #50 years makes slr right with scaled slr by e-4, 100 years makes it wrong, 0.1 init slr makes slr right
#1000 init slr makes it wrong.
# check_env(env, warn=True)

#train with A2C
model = PPO('MlpPolicy', env, verbose=1).learn(5000)
obs = env.reset()
#output = np.zeros(shape = (100, 4))

n_steps = 100
cost = []
sea_level = []
rewards = []
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  cost.append(obs[1])
  sea_level.append(obs[2])
  rewards.append(reward)
  print('obs', obs, 'reward=', reward, 'done=', done)
  # output[step] = [step, obs, action, reward]
  env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break

#plotting
# plt.plot(cost)
# plt.title("Cost trajectory of sea level rise over 100 years")
# plt.plot(sea_level)
# plt.title("Sea level rise trajectory over 100 years")
plt.plot(rewards)
plt.title("Reward trajectory over 100 years")
plt.show()

#stepping through environment randomly
# env.render()

# print(env.observation_space)
# print(env.action_space)
# print(env.action_space.sample())

# action = np.random.randint(0,3)
# n_steps = 20
# for step in range(n_steps):
#   print("Step {}".format(step + 1))
#   info, reward, done = env.step(action)
#   print('info', info, 'reward=', reward, 'done=', done)
#   env.render()
#   if done:
#     print("Goal reached!", "reward=", reward)
#     break
