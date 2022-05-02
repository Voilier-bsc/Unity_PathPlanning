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




#파라미터 값 세팅 
visual_state_size = [3, 256, 256]
vector_state_size = 7 #agent position, target position, relative distance
action_size = 2


load_model = False
train_mode = True

batch_size = 128
mem_maxlen = 50000  #replay memory 크기

discount_factor = 0.9
actor_lr = 1e-4
critic_lr = 5e-4
tau = 1e-3          #soft target update를 위한 파라미터


# OU noise paremeters
mu = 0              # 회귀할 평균의 값
theta = 1e-3        # 얼마나 평균으로 빨리 회귀할 지 결정하는 값
sigma = 2e-3        # 랜덤 프로세스의 변동성

run_step = 50000 if train_mode else 0
test_step = 10000
train_start_step = 5000

print_interval = 10
save_interval = 100

VISUAL_OBS = 0
VECTOR_OBS = 1

# 유니티 환경 경로 설정
game = "Single_Agent_RL"
env_name = "Single_Agent_RL"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S")
save_path = f"./saved_models/A2C/{date_time}"
load_path = f"./saved_models/A2C/2022_0316_1022_03"


# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OU_noise class -> ou noise정의 및 파라미터 결정
class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = np.ones((1, action_size), dtype=np.float32) * mu
    
    def sample(self):
        dx = theta * (mu - self.X) + sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X


class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        ## visual obs
        self.conv1 = torch.nn.Conv2d(in_channels=visual_state_size[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((visual_state_size[1] - 8)//4 + 1, (visual_state_size[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)

        self.flat = torch.nn.Flatten()
        self.fc1_visual = torch.nn.Linear(64*dim3[0]*dim3[1], 128)



        ## vector obs
        self.fc1_vector = torch.nn.Linear(vector_state_size, 128)
        self.fc2_vector = torch.nn.Linear(128,128)


        ## concat 이후
        self.fc1 = torch.nn.Linear(256,128)
        self.fc2 = torch.nn.Linear(128,128)
        self.mu = torch.nn.Linear(128, action_size)


    def forward(self, visual_state, vector_state):

        #VIUSAL
        visual_x = visual_state.permute(0, 3, 1, 2)
        visual_x = F.relu(self.conv1(visual_x))
        visual_x = F.relu(self.conv2(visual_x))
        visual_x = F.relu(self.conv3(visual_x))
        visual_x = self.flat(visual_x)
        visual_x = F.relu(self.fc1_visual(visual_x))

        #VECTOR
        vector_x = F.relu(self.fc1_vector(vector_state))

        #CONCAT
        x = torch.cat([visual_x,vector_x], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.tanh(self.mu(x))
            
class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        ## visual obs
        self.conv1 = torch.nn.Conv2d(in_channels=visual_state_size[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((visual_state_size[1] - 8)//4 + 1, (visual_state_size[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)

        self.flat = torch.nn.Flatten()
        self.fc1_visual = torch.nn.Linear(64*dim3[0]*dim3[1], 128)


        ## vector obs
        self.fc1_vector = torch.nn.Linear(vector_state_size, 128)
        self.fc2_vector = torch.nn.Linear(128,128)


        ## concat 이후
        self.fc1 = torch.nn.Linear(258,128)
        self.q = torch.nn.Linear(128,1)
    
    def forward(self, visual_state, vector_state, action):
        #VIUSAL
        visual_x = visual_state.permute(0, 3, 1, 2)
        visual_x = F.relu(self.conv1(visual_x))
        visual_x = F.relu(self.conv2(visual_x))
        visual_x = F.relu(self.conv3(visual_x))
        visual_x = self.flat(visual_x)
        visual_x = F.relu(self.fc1_visual(visual_x))

        #VECTOR
        vector_x = F.relu(self.fc1_vector(vector_state))

        #CONCAT
        x = torch.cat([visual_x,vector_x,action], dim=-1) # 128 + 128 + 2
        x = torch.relu(self.fc1(x))
        return self.q(x)

class DDPGAgent():
    def __init__(self):
        self.actor = Actor().to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)

        self.critic = Critic().to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)

        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.target_actor.load_state_dict(checkpoint["actor"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.target_critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def get_action(self, visual_state, vector_state, training=True):
        #network mode 설정
        self.actor.train(training)
        
        action = self.actor(torch.FloatTensor(visual_state).to(device), torch.FloatTensor(vector_state).to(device)).cpu().detach().numpy()
        return action + self.OU.sample() if training else action
        
    ## replay memory에 데이터 추가
    def append_sample(self, visual_state, vector_state, action, reward, next_visual_state, next_vector_state, done):
        self.memory.append((visual_state, vector_state, action, reward, next_visual_state, next_vector_state, done))


    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        visual_state        = np.stack([b[0] for b in batch], axis=0)
        vector_state        = np.stack([b[1] for b in batch], axis=0)
        action              = np.stack([b[2] for b in batch], axis=0)
        reward              = np.stack([b[3] for b in batch], axis=0)
        next_visual_state   = np.stack([b[4] for b in batch], axis=0)
        next_vector_state   = np.stack([b[5] for b in batch], axis=0)
        done                = np.stack([b[6] for b in batch], axis=0)

        visual_state, vector_state, action, reward, next_visual_state, next_vector_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                                            [visual_state, vector_state, action, reward, next_visual_state, next_vector_state, done])

        # Critic update
        next_actions = self.target_actor(next_visual_state, next_vector_state)
        next_q = self.target_critic(next_visual_state, next_vector_state, next_actions)
        target_q = reward + (1 - done) * discount_factor * next_q
        q = self.critic(visual_state, vector_state, action)
        critic_loss = F.mse_loss(target_q, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        action_pred = self.actor(next_visual_state, next_vector_state)
        actor_loss = -self.critic(next_visual_state, next_vector_state, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    # soft target update
    def soft_update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor" : self.actor.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "critic" : self.critic.state_dict(),
            "critic_optimizer" : self.critic_optimizer.state_dict(),
        }, save_path+'/ckpt')

    # 학습 기록
    def write_summray(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)

engine_configuration_channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel])
env.reset()

# 유니티 브레인 설정
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]
engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
dec, term = env.get_steps(behavior_name)

# DDPGAgent 클래스를 agent로 정의
agent = DDPGAgent()

actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0

for step in range(run_step + test_step):
    if step == run_step:
        if train_mode:
            agent.save_model()
        print("TEST START")
        train_mode = False
        engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

    visual_state = dec.obs[VISUAL_OBS]
    vector_state = dec.obs[VECTOR_OBS]
    
    action = agent.get_action(visual_state, vector_state, train_mode)
    action_tuple = ActionTuple()
    action_tuple.add_continuous(action)
    env.set_actions(behavior_name, action_tuple)
    env.step()

    dec, term = env.get_steps(behavior_name)
    done = len(term.agent_id) > 0
    reward = term.reward if done else dec.reward
    next_visual_state = term.obs[0] if done else dec.obs[VISUAL_OBS]
    next_vector_state = term.obs[0] if done else dec.obs[VECTOR_OBS]
    score += reward[0]

    if train_mode:
        agent.append_sample(visual_state[0], vector_state[0], action[0], reward, next_visual_state[0], next_vector_state[0], [done])
        
    if train_mode and step > max(batch_size, train_start_step):
        # train 수행
        actor_loss, critic_loss = agent.train_model()
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        # target network soft update
        agent.soft_update_target()

    if done:
        episode += 1
        scores.append(score)
        score = 0

        ## tensorboard
        if episode % print_interval == 0:
            mean_score = np.mean(scores)
            mean_actor_loss = np.mean(actor_losses)
            mean_critic_loss = np.mean(critic_losses)
            agent.write_summray(mean_score, mean_actor_loss, mean_critic_loss, step)
            actor_losses, critic_losses, scores = [], [], []

            print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                    f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

        # 네트워크 모델 저장
        if train_mode and episode % save_interval == 0:
            agent.save_model()

env.close()