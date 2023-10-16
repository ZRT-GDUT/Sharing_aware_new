from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

import model_util


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, num_states, num_action):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 256)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(128, num_action)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN:
    def __init__(self, num_state, num_action, gamma=0.9, batch_size=128, q_network_iteration=100, lr=0.01, memory_capacity=2000, episilo=0.9):
        self.eval_net, self.target_net = Net(num_state, num_action), Net(num_state, num_action)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.num_state = num_state
        self.num_action = num_action
        self.q_network_iteration = q_network_iteration
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((self.memory_capacity, self.num_state * 2 + 2))
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.episilo = episilo

    def choose_action(self, state, finished=False, intersection_list=None, rsu_num=0):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if intersection_list is not None:
            action_value = self.eval_net.forward(state)
            max_action_value = -10000000
            for idx in intersection_list:
                if action_value[0, idx].item() > max_action_value:
                    max_action_value = action_value[0, idx].item()
                    max_action_value_idx = idx
            src_rsu = int(max_action_value_idx / (rsu_num * len(model_util.Sub_Model_Structure_Size)))
            return src_rsu
        if np.random.randn() <= self.episilo:
            action_value = self.eval_net.forward(state)
            if finished:
                return action_value[0, 300].item()
            action = torch.max(action_value, 1)[1].data.item()
        else:
            action_value = self.eval_net.forward(state)
            if finished:
                return action_value[0, 300].item()
            action = np.random.randint(0, self.num_action)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        #update the parameters
        if self.learn_step_counter % self.memory_capacity ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.num_state])
        batch_action = torch.LongTensor(batch_memory[:, self.num_state:self.num_state+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.num_state+1:self.num_state+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-self.num_state:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

