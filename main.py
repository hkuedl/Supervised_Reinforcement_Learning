import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import os
from datetime import datetime
from Env.Env import EH_Model
import torch
from torch.utils.tensorboard import SummaryWriter
# torch.set_default_device("cuda:0")
import time
import copy
import torch.optim.lr_scheduler as lr_scheduler

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from imitation.algorithms import bc
from imitation.data.types import Transitions



class TrainRewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.writer: SummaryWriter = None
        self.episode_reward = 0.0
        self.episode_count = 1

    def _on_training_start(self) -> None:
        # logdir = self.logger.dir or "./ppo_tensorboard"
        logdir = "/home/user/workspaces/SRL/logs/episode_reward/"                    # input your log directory name here
        self.writer = SummaryWriter(logdir)

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]  # single env
        done = self.locals["dones"][0]
        self.episode_reward += reward
        if done:
            self.writer.add_scalar("train/episode_reward", self.episode_reward, self.episode_count)
            self.episode_reward = 0.0
            self.episode_count += 1
        return True

    def _on_training_end(self) -> None:
        self.writer.close()


class PrefillCallback(BaseCallback):
    def __init__(self, prefill_data, verbose=0):
        super().__init__(verbose)
        self.prefill_data = prefill_data # 一个包含所有预填充数据的字典
        self.prefilled = False  # 标记是否已经预填充

    def _on_training_start(self) -> None:
        # 在训练开始时，清空buffer并填入我们的数据
        self.model.rollout_buffer.reset()
        
        # 获取数据
        observations = self.prefill_data['observations']
        actions = self.prefill_data['actions']
        rewards = self.prefill_data['rewards']
        terminals = self.prefill_data['terminals']
        
        num_prefill_steps = observations.shape[0]
        
        # 计算values和log_probs
        import torch
        device = self.model.device
        values = []
        log_probs = []
        
        with torch.no_grad():
            for i in range(num_prefill_steps):
                obs_tensor = torch.FloatTensor(observations[i]).to(device)
                action_tensor = torch.FloatTensor(actions[i]).to(device)
                
                # 计算value - 保持为tensor
                value = self.model.policy.predict_values(obs_tensor.unsqueeze(0))
                values.append(value.flatten())  # 保持为tensor
                
                # 计算log_prob - 保持为tensor
                distribution = self.model.policy.get_distribution(obs_tensor.unsqueeze(0))
                log_prob = distribution.log_prob(action_tensor.unsqueeze(0))
                log_probs.append(log_prob.sum(axis=-1))  # 保持为tensor
        
        # 处理episode_start标记 (terminals的逻辑相反)
        episode_starts = np.zeros_like(terminals, dtype=bool)
        episode_starts[0] = True  # 第一个observation是episode的开始
        for i in range(1, len(terminals)):
            if terminals[i-1]:  # 如果前一步是terminal，那么这一步是新episode的开始
                episode_starts[i] = True
        
        # 循环填充buffer
        for i in range(num_prefill_steps):
            self.model.rollout_buffer.add(
                obs=observations[i],
                action=actions[i],
                reward=rewards[i],
                episode_start=episode_starts[i],
                value=values[i],
                log_prob=log_probs[i]
            )
        
        # 计算returns和advantage
        # 需要最后一个value来计算returns
        last_obs = torch.FloatTensor(observations[-1]).to(device)
        with torch.no_grad():
            last_values = self.model.policy.predict_values(last_obs.unsqueeze(0))
            last_values = last_values.flatten()  # 保持为tensor
        
        self.model.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, 
            dones=terminals[-1]
        )
        
        self.prefilled = True
        if self.verbose:
            print(f"✅ Prefilled buffer with {num_prefill_steps} expert transitions")
            print(f"   Mean reward: {np.mean(rewards):.3f}")
            print(f"   Mean value: {torch.stack(values).mean().item():.3f}")
            print(f"   Episodes: {np.sum(episode_starts)}")
            
            # 验证预填充是否成功
            # print("\n开始验证预填充结果...")
            # verify_prefill_success(self.model, self.prefill_data, verbose=True)
        
        return True

    def _on_step(self) -> bool:
        # BaseCallback要求实现这个方法，但我们的预填充只在训练开始时执行一次
        # 所以这里只需要返回True继续训练
        return True


def linear_to_floor(progress_remaining: float) -> float:
    initial_lr = 3e-4
    final_lr   = 5e-5
    return final_lr + (initial_lr - final_lr) * progress_remaining


# 设置训练
def train_ppo_agent(input_model, train_episodes, log_dir, copy_value_net=False, freeze = False, prefill_data=None, real_env=True, sim_train=False, freq=None, cut_eps=None):
    # 设置cuda
    # device = "cuda:0"
    
    # 创建环境
    env = EH_Model(output = log_dir, real_env=real_env, sim_train=sim_train, freq=freq, cut_eps=cut_eps)
    
    # 封装环境以便记录训练指标
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
       
    reward_logger = TrainRewardLogger()

    policy_kwargs = dict(log_std_init=-1,)

    # 初始化PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=linear_to_floor,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
    )

    # 用 BC 初始化 PPO
    ## 完整copy policy
    # model.policy.load_state_dict(input_model.policy.state_dict())

    ## 只copy policy 的policy net, action net, value nets
    model.policy.mlp_extractor.policy_net.load_state_dict(input_model.policy.mlp_extractor.policy_net.state_dict())
    model.policy.action_net.load_state_dict(input_model.policy.action_net.state_dict())
    model.policy.mlp_extractor.value_net.load_state_dict(input_model.policy.mlp_extractor.value_net.state_dict())
    model.policy.value_net.load_state_dict(input_model.policy.value_net.state_dict())

    if copy_value_net:
        model.policy.mlp_extractor.value_net.load_state_dict(model.policy.mlp_extractor.policy_net.state_dict(), strict=False)
    
    # 训练模型
    timesteps = train_episodes

    if prefill_data is not None:
        prefill_callback = PrefillCallback(prefill_data, verbose=1)
        model.learn(
            total_timesteps=timesteps,
            callback=[reward_logger, prefill_callback],
            tb_log_name=f"PPO_run",
            reset_num_timesteps=False
        )
    else:
        model.learn(
            total_timesteps=timesteps,
            callback=[reward_logger],
            tb_log_name=f"PPO_run"
        )

    # 保存最终模型
    model.save(f"{log_dir}/final_model")
    # np.save(f"{log_dir}/reward_curve.npy", np.array(env.envs[0].episode_reward_curve))
    
    print(f"训练完成，模型已保存到 {log_dir}/final_model")
    
    return model

def imitation_learning(act_dir, train_episodes=100, summer=True, freq=None):
    imitation_days = 30
    if summer:
        eval_days = 125
    else:
        eval_days = 334

    # actions comes from MILP                                     
    acts = np.load(act_dir).reshape((-1,3))
    acts[:,0] = acts[:,0] / 300
    acts[:,1] = (acts[:,1] * -1) /50
    acts[:,2] = acts[:,2] / 5600

    # # test the states and reward from MILP, whether it is the same as the env output, and get the dones and reward.
    S = []
    n_S = []
    R = []
    D = []
    env = EH_Model(real_env=False, freq = freq)
    states, _ = env.reset()
    S.append(copy.deepcopy(states))
    for i in range(imitation_days*24-1):
        states, reward, done, truncated, info = env.step(acts[i])
        n_S.append(copy.deepcopy(states))
        if done:
            states, _ = env.reset()
        S.append(copy.deepcopy(states))
        R.append(copy.deepcopy(reward))
        D.append(copy.deepcopy(done))
    states, reward, done, truncated, info = env.step(acts[i+1])
    R.append(copy.deepcopy(reward))
    D.append(copy.deepcopy(done))
    n_S.append(copy.deepcopy(states))
    S = np.array(S)
    n_S = np.array(n_S)
    R = np.array(R)
    D = np.array(D)

    obs = S
    next_obs = n_S
    dones = D
    rews = R
    infos = [True] * obs.shape[0]

    transitions = Transitions(obs, acts, infos, next_obs, dones)
    
    # 2. 训练 BC
    rng = np.random.default_rng(0)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        policy=PPO("MlpPolicy", env).policy,
    )
    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, eval_days)
    print(f"Reward before training: {reward_before_training}")

    policy = bc_trainer.policy
    print("policy before training",  next(policy.mlp_extractor.policy_net.parameters()).mean(), next(policy.mlp_extractor.value_net.parameters()).mean())  # 输出类似：ActorCriticPolicy(actor=..., critic=...)

    bc_trainer.train(n_epochs=train_episodes)

    policy = bc_trainer.policy
    print("policy after training",  next(policy.mlp_extractor.policy_net.parameters()).mean(), next(policy.mlp_extractor.value_net.parameters()).mean())  # 输出类似：ActorCriticPolicy(actor=..., critic=...)

    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, eval_days)
    print(f"Reward after training: {reward_after_training}")

    return bc_trainer, obs, acts, rews, next_obs, dones


if __name__ == "__main__":
        # 记录开始时间
    start_time = time.time()

    model_name = "test"                                             # input your model name here
    output_dir = '/home/user/workspaces/SRL/new_logs/'              # input your output directory name here
    for i in range(1):
        ################################# hyperparameters setting
        real_env = True
        freq = None
        log_name = model_name+"v"+str(i+1)

        log_dir = output_dir+log_name

        os.makedirs(log_dir, exist_ok=True)
        
        ################################# imitation learning
        # 获得supervised trained model
        act_dir = "/home/user/workspaces/SRL/test_act"+"/his_Action.npy"         
        bc_trainer, states, actions, rewards, next_states, terminals = imitation_learning(act_dir, train_episodes=10, summer=True, freq=freq)

        ################################# 训练智能体
        train_episodes_sim = 720
        train_episodes_real = 720

        ########################################################### 使用仿真环境微调，然后再真实环境微调的性能
        ### 仿真环境Fine-tune
        model_sim = train_ppo_agent(bc_trainer, train_episodes_sim, log_dir = log_dir+'/sim_env', copy_value_net=False, freeze=False, real_env=False, sim_train=True, freq = freq)
        
        ### 真实环境Fine-tune
        model_final = train_ppo_agent(model_sim, train_episodes_real, log_dir = log_dir+'/real_env', copy_value_net=False, freeze=False, real_env=True , sim_train=False, freq = freq)

        training_time_seconds = time.time() - start_time
        training_time_hours = training_time_seconds / 3600  # 1小时 = 3600秒

        print('DONE!')



