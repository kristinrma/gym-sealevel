import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class SealevelEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, init_slr, years, max_spending):
        super(SealevelEnv, self).__init__()
        self.init_slr = init_slr #sealevel in current year
        self.years = years #number of years to run environment
        self.max_spending = max_spending
        self.curr_slr = self.init_slr
        self.curr_year = 0
        self.curr_spending = 0
        #maybe add discount rate attribute needed for optimal policy

        max_slr = 250 #sea level rise by 2100 with the highest greenhouse gas emissions according to climate.gov
        low = np.array([0, 0, self.init_slr]) 
        high = ([self.years, self.max_spending, max_slr]) #(year, spending, sea level rise)
        n_actions = 4 #0 for do nothing, 1 for protect, 2 for retreat, 3 for mitigate

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
       
    def step(self, action):
        if action == 0:
            self.curr_spending += 0 # no cost to doing nothing
            # self.curr_slr += 
        if action == 1:
            # self.curr_spending += 
            # self.curr_slr +=
        if action == 2:
            # self.curr_spending +=
            # self.curr_slr +=
        if action == 3:
            # self.curr_spending +=
            # self.curr_slr +=
        self.curr_year += 1
        done = bool(self.curr_slr >= max_slr 
        or self.curr_year == self.years 
        or self.curr_spending >= self.max_spending)
        reward = (1 - self.curr_slr / max_slr) + (1 -  self.curr_spending / max_spending)
        return np.array(self.curr_year, self.curr_spending, self.curr_slr), reward, done, {}
    def reset(self):
        self.curr_slr = self.init_slr
        self.curr_year = 0
        self.curr_spending = 0
        return np.array(self.curr_year, self.curr_spending, self.curr_slr)
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
