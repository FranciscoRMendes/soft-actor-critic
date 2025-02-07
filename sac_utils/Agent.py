import torch as T
import torch.nn.functional as F
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
import numpy as np

# https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf# https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf

class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.pi_phi = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                                   name='actor', max_action=env.action_space.high)
        self.Q_theta_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                       name='critic_1')
        self.Q_theta_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                       name='critic_2')
        self.V_psi = ValueNetwork(beta, input_dims, name='value')
        self.V_psi_bar = ValueNetwork(beta, input_dims, name='target_value')
        self.scale = reward_scale
        self.update_psi_bar_using_psi(tau=1)

    @staticmethod
    def make_numpy(x):
        x = np.array(x.cpu().detach().numpy())
        return x

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.pi_phi.device)
        actions, _ = self.pi_phi.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_psi_bar_using_psi(self, tau=None):
        # This function corresponds to the update step inside algorithm 1
        # this is the last line in the algorithm
        # psi_bar = tau* psi + (1-tau)*psi_bar
        if tau is None:
            tau = self.tau

        psi_bar = self.V_psi_bar.named_parameters()
        psi = self.V_psi.named_parameters()

        target_value_state_dict = dict(psi_bar)
        value_state_dict = dict(psi)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()

        self.V_psi_bar.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.pi_phi.save_checkpoint()
        self.V_psi.save_checkpoint()
        self.V_psi_bar.save_checkpoint()
        self.Q_theta_1.save_checkpoint()
        self.Q_theta_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.pi_phi.load_checkpoint()
        self.V_psi.load_checkpoint()
        self.V_psi_bar.load_checkpoint()
        self.Q_theta_1.load_checkpoint()
        self.Q_theta_2.load_checkpoint()

    @staticmethod
    def process_sample(sample, device, dtype=T.float):
        """
        Converts a batch of sampled data into tensors and moves them to the specified device.

        Args:
            sample (tuple): A tuple containing (s_t, replay_buffer_a_t, r_t, s_t_plus_1, done).
            device (torch.device): The device to which the tensors should be moved.
            dtype (torch.dtype, optional): The data type of the tensors (except `done`). Defaults to torch.float.

        Returns:
            tuple: Processed tensors (s_t, replay_buffer_a_t, r_t, s_t_plus_1, done).
        """
        s_t, replay_buffer_a_t, r_t, s_t_plus_1, done = sample
        return (
            T.tensor(s_t, dtype=dtype).to(device),
            T.tensor(replay_buffer_a_t, dtype=dtype).to(device),
            T.tensor(r_t, dtype=dtype).to(device),
            T.tensor(s_t_plus_1, dtype=dtype).to(device),
            T.tensor(done, dtype=T.bool).to(device)  # Keeping `done` as a boolean tensor
        )

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # In these lines you just sample a batch of data from the replay buffer.
        # I use the word replay_buffer_a_t to differentiate from sampled_a_t, which is the action sampled from the actor network
        # the first is what actually happened in the environment, the second is what the actor network would sample given the state s_t

        sample = self.memory.sample_buffer(self.batch_size)
        s_t, a_t_rb, r_t, s_t_plus_1, done = self.process_sample(sample, self.pi_phi.device)


        V_psi_s_t = self.V_psi(s_t).view(-1)

        V_psi_bar_s_t_plus_1 = self.V_psi_bar(s_t_plus_1).view(-1)

        V_psi_bar_s_t_plus_1[done] = 0.0

        # In this section we take a sample from the actor network without the re-parameterization trick
        # we can use this because we do not need to back propagate through the actor network
        a_t_D, log_pi_t_D = self.pi_phi.sample_normal(s_t, reparameterize=False)


        log_pi_t_D = log_pi_t_D.view(-1)
        Q_theta_1_s_t_a_t_D = self.Q_theta_1.forward(s_t, a_t_D)
        Q_theta_2_s_t_a_t_D = self.Q_theta_2.forward(s_t, a_t_D)


        Q_theta_min_s_t_a_t_D = T.min(Q_theta_1_s_t_a_t_D, Q_theta_2_s_t_a_t_D)
        Q_theta_min_s_t_a_t_D = Q_theta_min_s_t_a_t_D.view(-1)

        ##############################################
        # VALUE NETWORK OPTIMIZATION
        # These lines below correspond to the loss function for the value network
        # This would be equation 5 in the paper
        ##############################################
        self.V_psi.optimizer.zero_grad()
        J_V_psi = 0.5 * F.mse_loss(V_psi_s_t, Q_theta_min_s_t_a_t_D - log_pi_t_D)
        J_V_psi.backward(retain_graph=True)
        self.V_psi.optimizer.step()


        ##############################################
        # ACTOR OR PI NETWORK OPTIMIZATION
        # In this section we take a sample from the actor network with re-parameterization trick
        # we can use this because we need to back propagate through the actor network
        ##############################################
        # a_t_D refers to actions drawn from a sample of the actor network and not the true actions taken from the replay buffer
        a_t_D, log_pi_t_D = self.pi_phi.sample_normal(s_t, reparameterize=True)
        log_pi_t_D = log_pi_t_D.view(-1)
        Q_theta_1_s_t_a_t_D = self.Q_theta_1.forward(s_t, a_t_D)
        Q_theta_2_s_t_a_t_D = self.Q_theta_2.forward(s_t, a_t_D)
        Q_theta_min_s_t_a_t_D = T.min(Q_theta_1_s_t_a_t_D, Q_theta_2_s_t_a_t_D)
        Q_theta_min_s_t_a_t_D = Q_theta_min_s_t_a_t_D.view(-1)

        # This is equation 12 in the paper
        # note that this is identical to the original loss function given by equation 10
        # after doing the re-parameterization trick
        J_pi_phi = T.mean(log_pi_t_D - Q_theta_min_s_t_a_t_D)
        self.pi_phi.optimizer.zero_grad()
        J_pi_phi.backward(retain_graph=True)
        self.pi_phi.optimizer.step()

        ################################################
        # CRITIC OR Q-NETWORK OPTIMIZATION
        ################################################
        # In this section we will optimize the two critic networks
        # We will use the bellman equation to calculate the target Q value
        self.Q_theta_1.optimizer.zero_grad()
        self.Q_theta_2.optimizer.zero_grad()
        # Equation 8 in the paper, in the paper the reward also depends on a_t
        # but in this case we get a constant reward for each step, so we can just use r_t
        # consequently, Q_hat_s_t AND NOT Q_hat_s_t_a_t
        Q_hat_s_t = self.scale*r_t + self.gamma*V_psi_bar_s_t_plus_1
        Q_theta_1_s_t_rb_at = self.Q_theta_1.forward(s_t, a_t_rb).view(-1) # this is the only place where actions from the replay buffer are used
        Q_theta_2_s_t_rb_at = self.Q_theta_2.forward(s_t, a_t_rb).view(-1)
        # this is equation 7 in the paper
        J_Q_theta_1_loss = 0.5 * F.mse_loss(Q_theta_1_s_t_rb_at, Q_hat_s_t)
        J_Q_theta_2_loss = 0.5 * F.mse_loss(Q_theta_2_s_t_rb_at, Q_hat_s_t)
        J_Q_theta_12 = J_Q_theta_1_loss + J_Q_theta_2_loss
        J_Q_theta_12.backward()
        self.Q_theta_1.optimizer.step()
        self.Q_theta_2.optimizer.step()

        self.update_psi_bar_using_psi()

        # actions_array = self.make_numpy(a_t_D)
        # value_array_ = np.array(V_psi_bar_s_t_plus_1.cpu().detach().numpy())
        # value_array = np.array(V_psi_s_t.cpu().detach().numpy())
        # q1_new_policy_arr = self.make_numpy(Q_theta_1_s_t_a_t_D)
        # q2_new_policy_arr = self.make_numpy(Q_theta_2_s_t_a_t_D)
        # critic_value_arr = self.make_numpy(Q_theta_min_s_t_a_t_D)
        # value_loss_arr = self.make_numpy(J_V_psi)


