import numpy as np
import torch
import torch.nn as nn
from ll_utils.networks import SoftQNetwork, ValueNetwork, PolicyNetwork
from ll_utils.utils import ReplayBuffer



class SoftActorCritic:
    def __init__(self, state_dim, action_dim, max_action, device, hidden_dim = 256):
        self.device = device
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)

        self.soft_q_net1 =  SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_optimizer_1 = torch.optim.Adam(self.soft_q_net1.parameters(), lr=3e-4)
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_optimizer_2 = torch.optim.Adam(self.soft_q_net2.parameters(), lr=3e-4)
        self.soft_q_criterion2 = nn.MSELoss()

        self.target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

        self.value_net = ValueNetwork(state_dim, hidden_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=3e-4)

        # self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.max_action = max_action
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
        self.device = device

        self.replay_buffer = ReplayBuffer(capacity=1000000)

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.policy_net.get_action(state).detach()

    def update(self, batch_size, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value = self.value_net(state)
        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)

        # Training Q Function
        target_value = self.target_value_net(next_state)
        # we update the two Q function param by reducing the MSE (minimum squared error) between the predicted Q value for a state-action pair and its corresponding target_q_value
        target_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        # print("Q Loss")
        # print(q_value_loss1)
        # clears gradient
        self.soft_q_optimizer_1.zero_grad()
        # passaggio di backward
        q_value_loss1.backward()
        # optimization step
        self.soft_q_optimizer_1.step()
        self.soft_q_optimizer_2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer_2.step()
        # Training Value Function
        # for the V network update we take the minimun of the two Q values
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        # substract from it the policy's log probability of selecting that action in that state
        target_value_func = predicted_new_q_value - log_prob
        # we decrese the MSE between the above quantity and the predicted V value of that state
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())
        # print("V Loss")
        # print(value_loss)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        # Training Policy Function
        # we update the policy by reducing the policy's log probability of choosing an action in a state log(pi(s)) - predicted Q-Value of that state-action pair

        policy_loss = (log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Here we use the Polyak for the target value network
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )