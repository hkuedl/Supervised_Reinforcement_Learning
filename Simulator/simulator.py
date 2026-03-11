import numpy as np
import torch
import time
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
matplotlib.use('Agg')


class Simulator:
    def __init__(self, agent, env, output, max_episode_length=24, summer=True, **env_args):
        self.agent = agent
        self.env = env
        self.output = output
        self.max_episode_length = max_episode_length

        #  Check the algorithm
        self.his_OBS = []
        self.his_ACTION = []
        self.his_TRADE = []
        self.his_COST = []
        self.his_COST_TUPLE = []
        self.his_REWARD = []

        self.reward_curve = []
        self.cost_curve = []

        if summer:
            self.train_days = 125   # 前4个月为训练集，最后一个月为测试集（共五个月：156天）
            self.total_days = 156

        self.env.current_step = self.train_days * 24


    def test(self):
        episode = self.train_days
        episode_reward = 0.
        episode_cost = 0.
        observation = None
        num_iterations = self.total_days - self.train_days

        for step in tqdm(range(num_iterations*24)):
            # reset if it is the start of episode
            if observation is None:
                observation, _ = deepcopy(self.env.reset())

            # agent pick action 
            action_raw, _ = self.agent.predict(observation, deterministic=True)
            action_raw = np.clip(action_raw, -1., 1.)            

            action_implementation = deepcopy(action_raw)          

            # for i in index:
            #     action_implementation[i] = (action_implementation[i] + 1) / 2

            # action correction and cost calculation
            action, trade, cost_tuple = self.env.cost_calculation(action_implementation)
            cost= -np.array(cost_tuple).sum()

            # env response with next_observation, reward, operation cost, and terminal flag (目前设定中reward=-opeartion_cost)
            observation2, reward, done, _, _ = self.env.step(action)       # oc_cost is a tuple of operation cost and carbon cost
            observation2 = deepcopy(observation2)


            # update 
            episode_reward += reward
            episode_cost += cost

            # write log
            self.logger(observation, action, trade, cost, reward, cost_tuple)
            observation = deepcopy(observation2)

            if done: # end of episode
                # reset
                observation = None
                self.reward_curve.append(episode_reward)
                self.cost_curve.append(episode_cost)
                episode_reward = 0.
                episode_cost = 0.
                episode += 1
                
        self.save_logger()

        total_cost = np.array(self.reward_curve).sum()

        return total_cost
    
    def logger(self, observation, action, trade, cost, reward, cost_tuple):  
        self.his_OBS.append(observation)
        self.his_ACTION.append(action)
        self.his_TRADE.append(trade)
        self.his_COST.append(cost)
        self.his_REWARD.append(reward)
        self.his_COST_TUPLE.append(cost_tuple)

        return None
    
    def save_logger(self):

        reward_curve = np.array(self.env.episode_reward_curve)
        oc_curve = np.array(self.env.episode_oc_curve)
        penalty_curve = np.array(self.env.episode_penalty_curve)

        np.save(self.output+'/his_obs.npy', np.array(self.his_OBS))
        np.save(self.output+'/his_action.npy', np.array(self.his_ACTION))
        np.save(self.output+'/his_trade.npy', np.array(self.his_TRADE))
        np.save(self.output+'/his_cost.npy', np.array(self.his_COST))
        np.save(self.output+'/his_reward.npy', np.array(self.his_REWARD))
        np.save(self.output+'/Episode_reward.npy', np.array(reward_curve))
        np.save(self.output+'/Episode_oc.npy', np.array(oc_curve))
        np.save(self.output+'/Episode_penalty.npy', np.array(penalty_curve))
        np.save(self.output+'/Episode_cost.npy', np.array(self.cost_curve))
        np.save(self.output+'/his_cost_tuple.npy', np.array(self.his_COST_TUPLE))
        
        return  None