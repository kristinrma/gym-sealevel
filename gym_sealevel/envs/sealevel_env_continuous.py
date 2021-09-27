import gym
from gym import error, spaces, utils
from gym.utils import seeding
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, A2C, PPO, SAC
# from stable_baselines.common.cmd_util import make_vec_env
import numpy as np
import matplotlib.pyplot as plt

class SealevelEnv_Continuous(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, init_slr, years, max_cost):
        super(SealevelEnv_Continuous, self).__init__()
        self.init_slr = 0 #sealevel in current year
        self.years = years #number of years to run environment
        self.max_cost = max_cost
        self.curr_slr = self.init_slr
        self.rand_slr = 0
        self.curr_year = 0
        self.curr_cost = 0.01
        self.r = 0.04 #maybe add discount rate attribute needed for optimal policy

        self.max_slr = 100 #sea level rise by 2100 with the highest greenhouse gas emissions according to climate.gov
        min_obs = np.array([0, 0, self.init_slr]) 
        max_obs = np.array([self.years, self.max_cost, self.max_slr]) #(year, cost, sea level rise)
        self.min_action = 0.0
        self.max_action = 1.0
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(min_obs, max_obs, dtype=np.float32)
       
    def step(self, action):
        self.rand_slr = np.random.uniform(0.0, 0.1)
        self.curr_cost += action[0]
        self.curr_slr += (1 - action[0])*self.rand_slr
        self.curr_year += 0.01
        done = bool(self.curr_slr >= self.max_slr 
        or self.curr_year == self.years 
        or self.curr_cost >= self.max_cost)
        reward = 1/(1/(1+self.r)**self.curr_year*self.curr_cost) #minimizing cost model
        # reward = 0
        # reward -= self.curr_cost
        return np.array([self.curr_year, self.curr_cost, self.curr_slr]), reward, done, {}
    
    def reset(self):
        self.curr_slr = self.init_slr
        self.curr_year = 0
        self.curr_cost = 0
        return np.array([self.curr_year, self.curr_cost, self.curr_slr])
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

env = SealevelEnv_Continuous(0.0, 1.0, 100)
#check_env(env, warn=True)

#train with an agent
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
  print("Action: ", action[0])
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
