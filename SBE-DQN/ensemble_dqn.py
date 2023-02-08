from collections import namedtuple

import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import sys
import matplotlib.pyplot as plt
Transition = namedtuple('Transition',['s','a','r','n_s','d'])

class ReplayBuffer(object):
	def __init__(self,buffer_size,batch_size):
		self.buffer = np.empty(buffer_size,dtype=Transition)
		self.buffer_size = buffer_size
		self.batch_size = batch_size
		self.pointer = 0
		self.can_sample = False
		self.is_full = False

	def push(self,transition):
		self.buffer[self.pointer] = transition
		self.pointer += 1 
		if self.pointer == self.batch_size:
			self.can_sample = True
		if self.pointer == self.buffer_size:
			self.is_full = True
			self.pointer = 0

	def sample(self):
		if self.is_full:
			return np.random.choice(self.buffer,self.batch_size)
		else:
			return np.random.choice(self.buffer[:self.pointer],self.batch_size)

	def debug(self):
		print(self.buffer)

class Mlp(nn.Module):
	def __init__(self,state_size,action_size,hidden_size):
		super(Mlp,self).__init__()
		self.fc1 = nn.Linear(in_features=state_size,out_features=hidden_size,bias=False)
		self.fc2 = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=False)
		self.fc3 = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=False)
		self.fc4 = nn.Linear(in_features=hidden_size,out_features=action_size,bias=False)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return self.fc4(x)


class EnsembleDQN(object):
	def __init__(self,state_size,action_size,hidden_size,batch_size,ensemble_size,device):
		self.gamma = 0.99
		self.device = device
		self.eps_train = 1.0
		self.eps_train_start = 1.0
		self.eps_train_end = 0.05
		self.eps_decay = 100
		self.eps_test = 0.0

		self.state_size = state_size
		self.action_size = action_size
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.buffer = ReplayBuffer(1000000,batch_size)

		#ensemble dqn
		self.ensemble_size = ensemble_size
		self.behavior_nets = \
		[Mlp(state_size, action_size, hidden_size).to(self.device) for _ in range(self.ensemble_size)]
		
		self.behavior_nets_ensembling = lambda x: \
		torch.tensor([self.behavior_nets[i](x).detach().numpy() for i in range(self.ensemble_size)]).sum(axis=0) / self.ensemble_size
		
		self.target_nets = \
		[Mlp(state_size, action_size, hidden_size).to(self.device) for _ in range(self.ensemble_size)]
		
		self.target_nets_ensembling = lambda x: \
		torch.tensor([self.target_nets[i](x).detach().numpy() for i in range(self.ensemble_size)]).sum(axis=0) / self.ensemble_size

		#optimizer
		self.optimizer_set = [torch.optim.Adam(self.behavior_nets[i].parameters(),lr=1e-3)\
		for i in range(self.ensemble_size)]
		

	def get_action(self,state,train=True):
		eps = self.eps_train*train + self.eps_test*(1-train)
		if np.random.random() < eps:
			action = torch.randint(0,self.action_size,size=(1,1))
		else:
			state = torch.from_numpy(state).float().squeeze()
			action = torch.argmax(self.behavior_nets_ensembling(state),-1,keepdim=True)
		action = action.item()
		return action

	def learn(self):
		transition = self.buffer.sample()
		state = torch.tensor([t.s for t in transition],dtype=torch.float32)
		action = torch.tensor([t.a for t in transition],dtype=torch.float32)
		reward = torch.tensor([[t.r] for t in transition])
		next_state = torch.tensor([t.n_s for t in transition],dtype=torch.float32)
		_done = torch.tensor(np.array([[1-t.d] for t in transition]))
		with torch.no_grad():
			next_q = self.target_nets_ensembling(next_state)
			target_q = reward + self.gamma*_done*next_q.max(1,keepdims=True).values

		action_selection = torch.eye(self.action_size)[action.view(-1).long()]

		for i in range(self.ensemble_size):
			q = (self.behavior_nets[i](state)*action_selection).sum(1,keepdims=True)
			loss = F.smooth_l1_loss(q,target_q)
			self.optimizer_set[i].zero_grad()
			loss.backward()
			self.optimizer_set[i].step()

	def store_transition(self,transition):
		self.buffer.push(transition)

	def can_sample(self):
		return self.buffer.can_sample

	def update_target(self):
		for i in range(self.ensemble_size):
			self.target_nets[i].load_state_dict(self.behavior_nets[i].state_dict())

	def epsilon_decay(self,steps):
		self.eps_train = self.eps_train_end + \
		(self.eps_train_start-self.eps_train_end)*np.exp(-1.*steps/self.eps_decay)
	
	def sample_q_value(self):
		transition = self.buffer.sample()
		state = torch.tensor([t.s for t in transition])
		ret = torch.tensor([self.behavior_nets[i](state).detach().numpy() for i in range(self.ensemble_size)]) / self.ensemble_size
		return ret

def seeding(seed,env,device):
	env.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device == 'cuda':
		torch.cuda.manual_seed_all(seed)

def test_model(agent,env,num_step):
	test_score = 0.0
	with torch.no_grad():
		state = env.reset()
		for _ in range(num_step):
			#env.render()
			action = agent.get_action(state,train=False)
			next_state, reward, done, _ = env.step(action)
			state = next_state if not done else env.reset()
			test_score += reward
			if done:
				break
	return test_score

def plot_cosine_similarity(agent, n_epi, seed, env_name, batch_size, ensemble_size):
	sampled_q = agent.sample_q_value()
	ret = torch.zeros(ensemble_size, ensemble_size)

	for i in range(batch_size):
		tmp_q = sampled_q[:, i, :]
		tmp_q = torch.squeeze(tmp_q)
		cosine_sim = F.cosine_similarity(tmp_q[:,:,None], tmp_q.t()[None,:,:])
		ret += cosine_sim
	ret = torch.div(ret , batch_size)
	plt.matshow(ret)
	plt.title('ensemble_dqn n_epi : {}'.format(n_epi + 1))
	plt.colorbar()
	plt.savefig("./{}/plot/ensemble_dqn_{}_seed{}.png".format(env_name, n_epi, seed))


if __name__ == '__main__':
	seed = 1
	env_name = 'CartPole-v1'
	#env_name = 'MountainCar-v0'
	print("run ensemble_dqn env_name : {} seed : {}".format(env_name, seed))
	nowDate = datetime.datetime.now()
	env = gym.make(env_name)
	#device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = 'cpu'
	seeding(seed,env,device)
	test_score_list = []
	
	num_episode = 500
	num_step = 1000
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	hidden_size = 60
	batch_size = 64
	ensemble_size = 10
	agent = EnsembleDQN(state_size,action_size,hidden_size,batch_size,ensemble_size,device)
	test_num = 5
	tot_step = 0

	for i_epi in range(num_episode):
		state = env.reset()
		train_score = 0.0
		for t in range(num_step):
			#env.render()
			action = agent.get_action(state)
			next_state, reward, done, _ = env.step(action)
			train_score += reward
			transition = Transition(state,action,reward,next_state,done)
			agent.store_transition(transition)
			state = next_state if not done else env.reset()
			tot_step += 1
			agent.epsilon_decay(tot_step)
			if agent.can_sample():
				agent.learn()
			if done:
				break
		if i_epi % 10 == 0:
			agent.update_target()
		if (i_epi + 1) % 100 == 0:
			plot_cosine_similarity(agent, i_epi, seed, env_name, batch_size, ensemble_size)

		test_result = 0
		for i in range(test_num):
			test_score = test_model(agent,env,num_step)
			test_result += test_score/test_num

		test_score_list.append(test_result)
		print("Episode {} test return: {:.5f}".format(i_epi,test_result))

	env.close()