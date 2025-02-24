import numpy as np
import torch
import torch.nn as nn
from ll_utils.networks import SoftQNetwork, ValueNetwork, PolicyNetwork
from ll_utils.utils import ReplayBuffer



class SoftActorCritic:
    def __init__(self, state_dim, action_dim, max_action, device, hidden_dim = 256):
        self.device = device
        self.pi_phi = PolicyNetwork(state_dim, action_dim, hidden_dim, device).to(device)
        self.policy_optimizer = torch.optim.Adam(self.pi_phi.parameters(), lr=3e-4)

        self.Q_theta_1 =  SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_optimizer_1 = torch.optim.Adam(self.Q_theta_1.parameters(), lr=3e-4)
        self.soft_q_criterion1 = nn.MSELoss()
        self.Q_theta_2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_optimizer_2 = torch.optim.Adam(self.Q_theta_2.parameters(), lr=3e-4)
        self.soft_q_criterion2 = nn.MSELoss()

        self.V_psi_bar = ValueNetwork(state_dim, hidden_dim).to(device)

        self.V_psi = ValueNetwork(state_dim, hidden_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.V_psi.parameters(), lr=3e-4)

        # self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.max_action = max_action
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
        self.device = device

        self.replay_buffer = ReplayBuffer(capacity=1000000)

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.pi_phi.get_action(state).detach()

    def update(self, batch_size, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        Q_theta_1_s_t_a_t_D  = self.Q_theta_1(state, action)
        Q_theta_2_s_t_a_t_D  = self.Q_theta_2(state, action)
        predicted_value = self.V_psi(state)
        new_action, log_prob, epsilon, mean, log_std = self.pi_phi.evaluate(state)

        #########################
        ## Training Q Function ##
        #########################
        target_value = self.V_psi_bar(next_state)
        # we update the two Q function param by reducing the MSE (minimum squared error) between the predicted Q value for a state-action pair and its corresponding target_q_value
        Q_hat_s_t_a_t  = reward + (1 - done) * gamma * target_value
        J_Q_theta_1_loss  = self.soft_q_criterion1(Q_theta_1_s_t_a_t_D , Q_hat_s_t_a_t .detach())
        J_Q_theta_2_loss  = self.soft_q_criterion2(Q_theta_2_s_t_a_t_D , Q_hat_s_t_a_t .detach())

        self.soft_q_optimizer_1.zero_grad()
        J_Q_theta_1_loss .backward()
        self.soft_q_optimizer_1.step()

        self.soft_q_optimizer_2.zero_grad()
        J_Q_theta_2_loss .backward()
        self.soft_q_optimizer_2.step()

        ###########################
        # Training Value Function #
        ###########################
        # for the V network we update using the minimum of the two Q values
        Q_theta_min_s_t_a_t_D  = torch.min(self.Q_theta_1(state, new_action), self.Q_theta_2(state, new_action))
        # substract from it the policy's log probability of selecting that action in that state
        target_value_func = Q_theta_min_s_t_a_t_D  - log_prob
        # we decrese the MSE between the above quantity and the predicted V value of that state
        J_V_psi = self.value_criterion(predicted_value, target_value_func.detach())

        self.value_optimizer.zero_grad()
        J_V_psi.backward()
        self.value_optimizer.step()

        ############################
        # Training Policy Function #
        ############################

        # Training Policy Function
        # we update the policy by reducing the policy's log probability of choosing an action in a state log(pi(s)) - predicted Q-Value of that state-action pair

        J_pi_phi = (log_prob - Q_theta_min_s_t_a_t_D ).mean()

        self.policy_optimizer.zero_grad()
        J_pi_phi.backward()
        self.policy_optimizer.step()

        # Here we use the Polyak for the target value network
        for target_param, param in zip(self.V_psi_bar.parameters(), self.V_psi.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def save(self, filename):
        """
        Saves all networks to the filename
        :param filename: file name to save
        :return: None
        """
        torch.save({
            'policy_net': self.pi_phi.state_dict(),
            'value_net': self.V_psi.state_dict(),
            'soft_q_net1': self.Q_theta_1.state_dict(),
            'soft_q_net2': self.Q_theta_2.state_dict(),
            'target_value_net': self.V_psi_bar.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'soft_q_optimizer_1': self.soft_q_optimizer_1.state_dict(),
            'soft_q_optimizer_2': self.soft_q_optimizer_2.state_dict()
        }, filename)

    @classmethod
    def from_file(cls, filename, state_dim, action_dim, max_action, device, hidden_dim=256):
        """
        Loads a SoftActorCritic model from a file.
        """
        model = cls(state_dim, action_dim, max_action, device, hidden_dim)
        checkpoint = torch.load(filename, map_location=device, weights_only=False)

        model.pi_phi.load_state_dict(checkpoint['policy_net'])
        model.V_psi.load_state_dict(checkpoint['value_net'])
        model.Q_theta_1.load_state_dict(checkpoint['soft_q_net1'])
        model.Q_theta_2.load_state_dict(checkpoint['soft_q_net2'])
        model.V_psi_bar.load_state_dict(checkpoint['target_value_net'])

        model.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        model.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        model.soft_q_optimizer_1.load_state_dict(checkpoint['soft_q_optimizer_1'])
        model.soft_q_optimizer_2.load_state_dict(checkpoint['soft_q_optimizer_2'])

        return model


