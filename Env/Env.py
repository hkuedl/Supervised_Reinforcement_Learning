import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from Env.Battery import BatteryStorage


class EH_Model(gym.Env):
    """
    Observation(Power):         
        Type: -
        Num     Observation                                    Min                     Max
        0       SoC of energy storage                           -                       -
        1       Real-time electricity price(buy)                -                       -
        2       Real-time electricity price(sell)               -                       -
        3       Predicted output power of PV                    -                       -
        4       Predicted output power of wind turbine          -                       -
        5       Energy demand                                   -                       -
        6       Indoor temperature                              -                       -
        7       Outdoor temperature                             -                       -
        8       Time                                            -                       - 

    Actions(Power):         
        Type: -
        Num       Action                                        Min                    Max
        0         Output of diesel generator                     0                      1
        1         Charging/discharging of energy storage        -1                      1      
        2         Power of TCL                                   0                      1      
    """
    def __init__(self, current_step =0, max_steps=24, test = False, output = None, real_env = True, summer= True, sim_train = False, freq=None, cut_eps = None):
        # super(EH_Model, self).__init__()
        # define the observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),  # SoC of energy storage, electricity price(buy), electricity price(sell), output power of PV, output power of wind turbine, energy demand, time
            high=np.array([1, 1, 1, 100, 200, 300, 50, 50, 24]),  # SoC of energy storage, electricity price(buy), electricity price(sell), output power of PV, output power of wind turbine, energy demand, time
            dtype=np.float64
        )
        
        # define the action space
        self.action_space = spaces.Box(
            low=np.array([0, -1, 0]),          # 柴油发电机功率的最小值, 储能充放电功率的最小值, tcl出力功率最小值
            high=np.array([1, 1, 1]),          # 柴油发电机功率的最大值, 储能充放电功率的最大值，tcl出力功率最大值
            dtype=np.float64
        )

        # environment parameters
        self.max_steps = max_steps
        self.test = test
        self.current_step = current_step 
        self.episode_reward_curve = []
        self.episode_oc_curve = []              # operation cost
        self.episode_penalty_curve = []
        self.episode_reward = 0
        self.episode_oc = 0
        self.episode_penalty = 0
        self.output = output
        self.real_env = real_env
        self.summer = summer
        self.sim_train = sim_train

        if self.summer and self.sim_train:
            self.train_days = 62    # 前两个月为仿真训练集，最后一个月为测试集（共三个月：156天）
            self.total_days = 156
        elif self.summer and not self.sim_train:
            self.train_days = 125   # 前4个月为训练集，最后一个月为测试集（共五个月：156天）
            self.total_days = 156
 
        
        ## Parameters setting
        self.std = 0.0                                # 环境不确定性，e.g., demand, RES
        self.action_dim = 3                           # action数量
        self.obs_dim = 9                              # observation数量
        self.seed = 0
        ### parameters of TCL
        self.comfortable_pen = 10
        self.R_th = 2.0/1000           # 热阻 (°C/W)
        self.C_th = 2.0*1000*3600      # 热容 (W·s/°C)
        self.eta = 2.5                 # 能效系数 (COP)
        self.P_rated = 5600            # 额定功率 (W)
        self.P_rated = 3000            # 额定功率 (W)
        self.theta_r = 22              # 设定温度 (°C)
        self.delta = 1.5                 # 死区宽度 (°C)
        self.a = 1 / (self.R_th * self.C_th)     #  (1/s)         
        self.b = self.eta / self.C_th            # (°C/(W·s))
        if self.real_env:
            self.freq = 1                  # 温度更新频率 (min)     
        else:
            if freq is None:
                self.freq = 60                 # 温度更新频率 (min)      
            else:
                self.freq = freq

        ### power max of energy devices 
        self.grid_max = 999                           # max power of grid import and export
        self.dg_power_max = 300                       # max power out of diesel generator
        self.es_power_max = 50                        # max power out of energy storage
        ### parameters of storages
        self.es_capacity = 200                        # energy capacity of energy storage
        self.es_efficiency = 0.95                     # 充放效率, ref:0.95
        if real_env:
            self.battery = BatteryStorage(E_ess=self.es_capacity, SoC_max=1.0, SoC_min=0, P_max=self.es_power_max, standby_loss=0.0, detailed = True)
        ### production cost of devices
        self.cost_dg = 0.20                           # 维护成本+可变成本（燃料）

        ## Data loading              !需要替换为本地的路径!
        if self.summer:
            energy_demand = pd.read_excel('/home/user/workspaces/Supervised-RL/Env/dataset_summer.xlsx', sheet_name = 0)
            elec_price = pd.read_excel('/home/user/workspaces/Supervised-RL/Env/dataset_summer.xlsx', sheet_name = 1)
            res = pd.read_excel('/home/user/workspaces/Supervised-RL/Env/dataset_summer.xlsx', sheet_name = 2)
            temp = pd.read_excel('/home/user/workspaces/Supervised-RL/Env/dataset_summer.xlsx', sheet_name = 4)


        ### Demand
        self.energy_demand = np.array(energy_demand).reshape((self.total_days,24,3))

        ### Temperature
        self.temp = np.delete(self.non_linear_interpolate(np.array(temp)), -1, axis=0).reshape((self.total_days,24*60,1))

        ### Real data
        self.energy_demand_real = copy.deepcopy(np.array(energy_demand)).reshape((self.total_days,24,3))
        self.res_real = copy.deepcopy(np.array(res))
        self.wind_real = self.res_real[:,0].reshape((self.total_days,24))
        self.pv_real = self.res_real[:,1].reshape((self.total_days,24))
        self.temp_real = copy.deepcopy(self.temp)

        ### Adding noise
        energy_demand = self.generate_normal_random_matrix(np.array(energy_demand), self.std)
        res = self.generate_normal_random_matrix(np.array(res), self.std)
        temp = self.generate_normal_random_matrix(np.array(temp), self.std)
    
        ### Real data with noise
        ### Noisy observation
        self.energy_demand = np.array(energy_demand).reshape((self.total_days,24,3))
        self.wind = res[:,0].reshape((self.total_days,24))
        self.pv = res[:,1].reshape((self.total_days,24))
        self.temp = np.delete(self.non_linear_interpolate(np.array(temp)), -1, axis=0).reshape((self.total_days,24*60,1))

        ### Price
        self.electricity_price = np.array(elec_price)
        
        if cut_eps is not None:
            self.energy_demand_real = self.energy_demand_real[cut_eps:self.total_days]
            self.energy_demand = self.energy_demand[cut_eps:self.total_days]
            self.wind = self.wind[cut_eps:self.total_days]
            self.pv = self.pv[cut_eps:self.total_days]
            self.temp = self.temp[cut_eps:self.total_days]
            
            self.train_days = self.train_days - cut_eps
            self.total_days = self.total_days - cut_eps

        # initialize the state
        self.reset()

    def step(self, action):

        # action correction
        safe_action, safe_trade, cost = self.cost_calculation(action)
        output_dg = safe_action[0] * self.dg_power_max
        output_es = safe_action[1] * self.es_power_max 
        output_tcl = safe_action[2] * self.P_rated

        # Reward calculation
        ## get real demand and RES output       
        energy_delta = safe_trade[0]              # 正值表明从电网购电，负值表明向电网售电

        if energy_delta >= 0:
            lambda_electricity = self.states[1]
        else:
            lambda_electricity = self.states[2]

        total_cost = (lambda_electricity * energy_delta) + (self.cost_dg * output_dg)

        reward = - total_cost / 15

        info = {
            'total_cost': total_cost,
        }

        # update the state
        ## update the SOC of energy storage
        self.states[0] = self.get_next_SOC(self.states[0], output_es)
        ## update the indoor temperture
        self.states[6] = self.get_next_temperature(self.states[6], output_tcl)
        ## update remaining states
        self._update_state()                                 # 更新SOC和indoor temp以外的状态

        env_step = self.current_step % self.max_steps
        done = env_step >= (self.max_steps - 1)

        self.current_step += 1                               # 更新时间步
    
        # Final reward calculation 
        ## Penalty calculation
        temp_violation = 0
        if self.states[6] > self.theta_r + self.delta:
            temp_violation = abs(self.states[6] - (self.theta_r + self.delta))
        elif self.states[6] < self.theta_r - self.delta:
            temp_violation = abs(self.states[6] - (self.theta_r - self.delta))
        total_penalty =  temp_violation * self.comfortable_pen


        # reward = -( total_cost + total_penalty) 

        truncated = False

        if self.output is not None:
            if self.current_step > 0 and self.current_step % 2000 ==0:
                self.plot_epr()                               # 每2000步画一次图

        if not done:
            self.episode_reward += reward
            self.episode_oc += (-total_cost)
            self.episode_penalty += (-total_penalty)
        else:
            self.episode_reward_curve.append(self.episode_reward)
            self.episode_oc_curve.append(self.episode_oc)
            self.episode_penalty_curve.append(self.episode_penalty)
            self.episode_reward = 0
            self.episode_oc = 0
            self.episode_penalty = 0

        return self.states, reward, done, truncated, info
    
    def _update_state(self):
        episode_step = self.current_step // self.max_steps
        env_step = self.current_step % self.max_steps
        if self.test:
            iteration = episode_step
        else:
            iteration = episode_step % self.train_days                       
        
        if env_step >= self.max_steps - 1:                                    # 每天都为一个轮次
            episode_done = True
        else:
            episode_done = False

        if not episode_done:
            self.states[1] = self.electricity_price[env_step+1] 
            self.states[2] = self.electricity_price[env_step+1] / 2                             #购价为售价的2倍
            self.states[3] = self.pv[iteration][env_step+1]
            self.states[4] = self.wind[iteration][env_step+1]
            self.states[5] = self.energy_demand[iteration][env_step+1].sum()
            self.states[7] = self.temp[iteration][(env_step+1)*60]                              # 室外温度
            self.states[8] = env_step + 1
        else:
            self.states[1] = 0 
            self.states[2] = 0 
            self.states[3] = 0 
            self.states[4] = 0 
            self.states[5] = 0 
            self.states[7] = 0 
            self.states[8] = 0 
        
        return copy.deepcopy(self.states)


    def reset(self, seed=None, options=None):
        if seed is not None:
            seed = seed
        else:
            seed = self.seed

        episode_step = self.current_step // self.max_steps
        if self.test:
            initialized_episode_step = episode_step
        else:
            initialized_episode_step = episode_step % self.train_days

        initialized_env_step = 0
        initial_states = np.zeros(self.obs_dim, dtype=np.float32)

        ### PSO initialized state
        np.random.seed(self.seed)
        initial_states[0] = 0
        initial_states[1] = self.electricity_price[initialized_env_step]             
        initial_states[2] = self.electricity_price[initialized_env_step] / 2                            #购价为售价的2倍
        initial_states[3] = self.pv[initialized_episode_step, initialized_env_step]
        initial_states[4] = self.wind[initialized_episode_step, initialized_env_step]
        initial_states[5] = self.energy_demand[initialized_episode_step, initialized_env_step].sum()
        initial_states[6] = self.theta_r
        initial_states[7] = self.temp[initialized_episode_step, initialized_env_step]                     # 室外温度
        initial_states[8] = initialized_env_step        

        self.states = initial_states

        info = {
            'seed': seed,
        }

        return copy.deepcopy(self.states), info

    def render(self, mode='human'):
        # 简单打印当前状态       
        print(f"Step: {self.current_step}")
        print("-" * 50)

    def close(self):
        pass

    def generate_normal_random_matrix(self, input_matrix, std):               # real value as mean, 10% of mean as std
        length = input_matrix.shape[0]
        width = input_matrix.shape[1]
        output_matrix = np.zeros((length, width))
        for i in range(length):
            for j in range(width):
                output_matrix[i,j] = np.random.normal(input_matrix[i,j], abs(std*input_matrix[i,j]))
        
        return output_matrix
    
    def cost_calculation(self, action):       # the input state is the real state

        episode_step = self.current_step // self.max_steps
        env_step = self.current_step % self.max_steps
        if self.test:
            episode_step = episode_step
        else:
            episode_step = episode_step % self.train_days                        
        

        # based on real state
        ## demand
        demand_now = self.energy_demand_real[episode_step][env_step].sum()                                # 真实负荷
        ## RES generation
        output_res = self.wind_real[episode_step][env_step] + self.pv_real[episode_step][env_step]        # 真实的风能和太阳能发电量
        ## SoC
        SOC_es = self.states[0]

        safe_action = copy.deepcopy(action)
        # current action    
        action_dg = action[0] * self.dg_power_max
        action_es = action[1] * self.es_power_max  
        action_tcl = action[2] * self.P_rated

        ## overcharging or overdischarging
        ### energy storage
        cost1 = 0
        maxp_storage_es = self.es_power_max
        capacity_storage_es = self.es_capacity
        efficiency_es = self.es_efficiency
        ubp_storage_es = min(capacity_storage_es * SOC_es * efficiency_es, maxp_storage_es)      
        lbp_storage_es = max((SOC_es - 1) * capacity_storage_es /efficiency_es, -maxp_storage_es)   
        if action_es <= ubp_storage_es and action_es >= lbp_storage_es:
            action_es = action_es
        elif action_es > ubp_storage_es:
            cost3 = action_es - ubp_storage_es
            action_es = ubp_storage_es
        elif action_es < lbp_storage_es:
            cost3 = lbp_storage_es - action_es
            action_es = lbp_storage_es
        safe_action[1] = action_es / maxp_storage_es

        
        # energy trading calculation
        action_upg = demand_now - action_dg - action_es - output_res + (abs(action_tcl) / 1000)
        
        
        # action correction & cost calculation
        ## electricity demand unbalance: i) import limit ii) export limit
        cost2 = 0
        res_curtailment = 0
        if action_upg > 0:
            cv_upg = action_upg - self.grid_max
            if cv_upg > 0:
                cost2 = cv_upg
                action_upg = self.grid_max
        if action_upg < 0:
            cv_upg =  abs(action_upg) - self.grid_max
            if cv_upg > 0 and (cv_upg-output_res) > 0:
                cost2 = cv_upg-output_res
                action_upg = - self.grid_max
                res_curtailment = output_res
            elif cv_upg > 0 and (cv_upg-output_res) < 0:
                action_upg = - self.grid_max
                res_curtailment = cv_upg
        
        ## temperature dead band
        cost3 = 0
        current_temp = self.states[6]
        next_temp = self.get_next_temperature(current_temp, action_tcl)
        if current_temp < (self.theta_r - self.delta):
            action_tcl = 0
            cost3 = abs(current_temp - (self.theta_r - self.delta))  # 温度过低的惩罚
        else:
            if next_temp < (self.theta_r - self.delta):
                if self.get_next_temperature(current_temp, 0) < (self.theta_r - self.delta):
                    action_tcl = 0
                    cost3 = abs(current_temp - (self.theta_r - self.delta))
                else:
                    # action_tcl修正为让温度刚好到达下限的功率
                    action_tcl = self.find_tcl_power_for_target_temperature(current_temp, self.theta_r - self.delta)
            elif next_temp > (self.theta_r + self.delta):
                if self.get_next_temperature(current_temp, self.P_rated) > (self.theta_r + self.delta):
                    action_tcl = self.P_rated
                    cost3 = abs(current_temp - (self.theta_r + self.delta))
                else:
                    # action_tcl修正为让温度刚好到达上限的功率
                    action_tcl = self.find_tcl_power_for_target_temperature(current_temp, self.theta_r + self.delta)
        safe_action[2] = action_tcl / self.P_rated

        safe_trade = np.array([action_upg, res_curtailment])
        cost = np.array([cost1, cost2, cost3])

        return safe_action, safe_trade, cost             # 这里返回的cost是约束违反的cost
    
    def plot_epr(self):
        plt.figure(figsize=(20,10))
        ep_reward = np.array(self.episode_reward_curve)
        x_plot = np.arange(len(self.episode_reward_curve))
        plt.plot(x_plot, ep_reward, linewidth=2, alpha=0.3, label='episode reward')
        epr_moving_average = self.moving_average(ep_reward, self.train_days)
        plt.plot(x_plot[:len(epr_moving_average)], np.array(epr_moving_average), linewidth=2, linestyle='--', color="aqua", label='moving average')
        plt.xlabel('Episode', fontsize =14)
        plt.ylabel('Reawrd', fontsize =14)
        plt.grid()
        plt.legend()
        plt.savefig(self.output+'/Reward.png')
        plt.show()
        plt.close()

    def moving_average(self, array, window):
        MA = []
        for i in range(array.shape[0] - window + 1):
            front = i
            tail = i + window
            MA.append(array[front : tail].mean())

        return MA

    def get_next_temperature(self, current_temperature, output_tcl):
        episode_step = self.current_step // self.max_steps
        env_step = self.current_step % self.max_steps
        if self.test:
            episode_step = episode_step
        else:
            episode_step = episode_step % self.train_days                        
        outdoor_temperature_list = self.temp[episode_step][env_step*60:(env_step+1)*60]  # 接下来一小时的分钟级别的室外温度

        p = output_tcl

        dt = self.freq*60                # 内部温度更新时间步长 (s)
        total_time = 1 * 3600            # 总时长 (1小时)
        steps = int(total_time / dt)

        current_theta = current_temperature        # 室内温度
        for i in range(0, steps):
            # 当前室外温度
            theta_a = outdoor_temperature_list[i*self.freq] 

            # 更新温度
            current_theta = self.update_temperature_euler(current_theta, theta_a, p, dt, self.a, self.b)
        next_temperature = current_theta
        return next_temperature


    def get_next_SOC(self, current_SOC, output_es):
        if self.real_env:
            self.battery.SoC = current_SOC
            current_SOC = self.battery.update_SoC(output_es, 1)               # 以output_es为充/放电功率，进行持续1小时的充/放电
        else:
            if output_es <= 0:                                                                                 #判断充放状态: 正值为放能，负值为充能
                current_SOC -= output_es / self.es_capacity * self.es_efficiency                            #delta t 默认为1h, 因为action是肯定保证不会让储能SOC超出范围[0,1]，因此这里不需要约束
            else:
                current_SOC -= output_es / self.es_capacity / self.es_efficiency

        next_SOC = current_SOC

        return next_SOC
    
    def update_temperature_euler(self, theta, theta_a, p, dt, a, b):
        """显式欧拉法更新温度"""
        dtheta_dt = -a * (theta - theta_a) - b * p
        theta_new = theta + dtheta_dt * dt
        return theta_new
    
    def find_tcl_power_for_target_temperature(self, current_temp, target_temp, tolerance=0.001):
        """
        使用二分法找到使下一时刻温度达到目标温度的TCL功率
        
        Args:
            current_temp: 当前温度
            target_temp: 目标温度
            tolerance: 温度误差容忍度
            
        Returns:
            最优TCL功率
        """
        # 设置功率搜索范围
        power_min = 0.0
        power_max = self.P_rated
        
        # 检查边界情况
        temp_at_min = self.get_next_temperature(current_temp, power_min)
        temp_at_max = self.get_next_temperature(current_temp, power_max)
        
        # 如果目标温度超出可达到的范围，返回边界值
        if target_temp >= temp_at_min:
            return power_min
        elif target_temp <= temp_at_max:
            return power_max
        
        # 二分法搜索
        max_iterations = 99
        for _ in range(max_iterations):
            power_mid = (power_min + power_max) / 2.0
            temp_at_mid = self.get_next_temperature(current_temp, power_mid)
            
            # 检查是否足够接近目标温度
            if abs(temp_at_mid - target_temp) < tolerance:
                return power_mid
            
            # 更新搜索范围
            if temp_at_mid > target_temp:
                # 温度太高，需要增加制冷功率
                power_min = power_mid
            else:
                # 温度太低，需要减少制冷功率
                power_max = power_mid
        
        # 如果达到最大迭代次数，进行报错
        raise ValueError("无法找到合适的TCL功率")
    
    def generate_ood_states(self, batch_states: np.ndarray, policy_actions: np.ndarray, 
                           batch_indices: np.ndarray = None) -> np.ndarray:
        """
        基于策略动作生成OOD状态，用于保守价值学习
        
        Args:
            batch_states: 当前状态批量 (batch_size, obs_dim)
            policy_actions: 策略生成的动作批量 (batch_size, action_dim)
            batch_indices: 批次索引（用于确定当前时间步）(batch_size,)
        
        Returns:
            ood_states: 生成的OOD状态 (batch_size, obs_dim)
        """
        batch_size = batch_states.shape[0]
        ood_states = np.zeros_like(batch_states)
        
        for i in range(batch_size):
            # 得到安全动作
            safe_action, _, _ = self.cost_calculation(policy_actions[i])

            # 设置环境状态为批次中的状态
            self.states = batch_states[i].copy()
            self.current_step = int(batch_indices[i])
            
            # 执行策略动作并获取下一状态
            next_state, _, _, _, _ = self.step(safe_action)
            
            ood_states[i] = next_state
        
        return ood_states

    def non_linear_interpolate(self, array):
        len_array = array.shape[0]
        x = np.linspace(0, 60*(len_array-1), len_array)
        spline = UnivariateSpline(x, array, s=0)  # s=0 表示精确插值（无平滑）
        x_new = np.linspace(0, 60*(len_array-1), 60*(len_array-1)+1)
        array_extend = spline(x_new)
    
        return array_extend
    

