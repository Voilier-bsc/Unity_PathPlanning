import numpy as np
import random
import copy
import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel
import config

config = config.configs()

#파라미터 값 세팅 

gpu_num           = config.gpu_num
no_graphics       = config.no_graphics
worker_id         = config.worker_id

visual_state_size = config.visual_state_size
vector_state_size = 47 # 42 + 5
action_size       = 9


load_model        = config.load_model
train_mode        = config.train_mode

batch_size        = config.batch_size 
mem_maxlen        = config.mem_maxlen

discount_factor   = config.discount_factor
actor_lr          = config.actor_lr
critic_lr         = config.critic_lr
tau               = config.tau 

learning_rate = 0.00025

# OU noise paremeters
mu                = config.mu
theta             = config.theta
sigma             = config.sigma

run_step          = config.run_step
test_step         = config.test_step
train_start_step  = config.train_start_step

print_interval    = config.print_interval
save_interval     = config.save_interval

VISUAL_OBS        = 0
VECTOR_OBS        = 2


# epsilon
epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = run_step * 0.8
eplsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0.



# 유니티 환경 경로 설정
game = "Single_Agent_RL_DQN"
# env_name = "../Unity_PathPlanning/Single_Agent_RL/Single_Agent_RL_DQN"
env_name = "../Single_Agent_RL_DQN/Single_Agent_RL_DQN"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S")
save_path = f"./saved_models/A2C/{date_time}"
load_path = f"./saved_models/A2C/2022_0504_1420_37"


# 연산 장치
device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")

class A2C(torch.nn.Module):
    def __init__(self, **kwargs):
        super(A2C, self).__init__(**kwargs)
        self.d1 = torch.nn.Linear(vector_state_size, 128)
        self.d2 = torch.nn.Linear(128, 128)
        self.pi = torch.nn.Linear(128, action_size)
        self.v = torch.nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return F.softmax(self.pi(x), dim=1), self.v(x)

# A2CAgent 클래스 -> A2C 알고리즘을 위한 다양한 함수 정의 
class A2CAgent:
    def __init__(self):
        self.a2c = A2C().to(device)
        self.optimizer = torch.optim.Adam(self.a2c.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.a2c.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # 정책을 통해 행동 결정 
    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.a2c.train(training)

        # 네트워크 연산에 따라 행동 결정
        pi, _ = self.a2c(torch.FloatTensor(state).to(device))
        action = torch.multinomial(pi, num_samples=1).cpu().numpy()
        return action

    # 학습 수행
    def train_model(self, state, action, reward, next_state, done):
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])
        pi, value = self.a2c(state)

        #가치신경망
        with torch.no_grad():
            _, next_value = self.a2c(next_state)
            target_value  = reward + (1-done) * discount_factor * next_value
        critic_loss = F.mse_loss(target_value, value)

        #정책신경망
        eye = torch.eye(action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        advantage = (target_value - value).detach()
        actor_loss = -(torch.log((one_hot_action * pi).sum(1))*advantage).mean()
        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

    # 네트워크 모델 저장
    def save_model(self, episode):
        path_name = "{}/ckpt_{}".format(save_path, episode)
        # print(f"... Save Model to {save_path}/ckpt ...")
        print(f"... Save Model to {path_name} ...")
        torch.save({
            "network" : self.a2c.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, path_name)

        # 학습 기록 
    def write_summray(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)


# 유니티 환경 경로 설정 (file_name)
engine_configuration_channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name=env_name,
                        side_channels=[engine_configuration_channel], no_graphics=False, worker_id=0)
env.reset()

# 유니티 브레인 설정 
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]
engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
dec, term = env.get_steps(behavior_name)

# A2C 클래스를 agent로 정의 
agent = A2CAgent()
actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
for step in range(run_step + test_step):
    if step == run_step:
        if train_mode:
            agent.save_model(episode)
        print("TEST START")
        train_mode = False
        engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

    state = np.concatenate([dec.obs[VISUAL_OBS], dec.obs[VECTOR_OBS]],axis=-1)
    action = agent.get_action(state, train_mode)
    real_action = action + 1
    action_tuple = ActionTuple()
    action_tuple.add_discrete(real_action)
    env.set_actions(behavior_name, action_tuple)
    env.step()

    #환경으로부터 얻는 정보
    dec, term = env.get_steps(behavior_name)
    done = len(term.agent_id) > 0
    reward = term.reward if done else dec.reward
    next_state = np.concatenate([term.obs[VISUAL_OBS], term.obs[VECTOR_OBS]],axis=-1)if done\
                    else np.concatenate([dec.obs[VISUAL_OBS], dec.obs[VECTOR_OBS]],axis=-1)
    score += reward[0]

    if train_mode:
        #학습수행
        actor_loss, critic_loss = agent.train_model(state, action[0], [reward], next_state, [done])
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

    if done:
        episode +=1
        scores.append(score)
        score = 0

        # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
        if episode % print_interval == 0:
            mean_score = np.mean(scores)
            mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
            mean_critic_loss = np.mean(critic_losses)  if len(critic_losses) > 0 else 0
            agent.write_summray(mean_score, mean_actor_loss, mean_critic_loss, step)
            actor_losses, critic_losses, scores = [], [], []

            print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                    f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

        # 네트워크 모델 저장 
        if train_mode and episode % save_interval == 0:
            agent.save_model(episode)
env.close()