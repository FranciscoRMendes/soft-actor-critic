import torch as T
import torch.nn.functional as F
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
import numpy as np

# https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf# https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf

class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2')
        self.value_net = ValueNetwork(beta, input_dims, name='value')
        self.target_value_net = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    @staticmethod
    def make_numpy(x):
        x = np.array(x.cpu().detach().numpy())
        return x

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value_net.named_parameters()
        value_params = self.value_net.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value_net.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value_net.save_checkpoint()
        self.target_value_net.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value_net.load_checkpoint()
        self.target_value_net.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

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
        s_t, replay_buffer_a_t, r_t, s_t_plus_1, done = self.process_sample(sample, self.actor.device)


        V_s_t = self.value_net(s_t).view(-1)

        V_s_t_plus_1 = self.target_value_net(s_t_plus_1).view(-1)

        V_s_t_plus_1[done] = 0.0

        sampled_a_t, sampled_log_pi_t = self.actor.sample_normal(s_t, reparameterize=False)


        sampled_log_pi_t = sampled_log_pi_t.view(-1)
        q_theta_1_new_policy = self.critic_1.forward(s_t, sampled_a_t)
        q_theta_2_new_policy = self.critic_2.forward(s_t, sampled_a_t)


        q_theta_min = T.min(q_theta_1_new_policy, q_theta_2_new_policy)
        q_theta_min = q_theta_min.view(-1)

        ##############################################
        # These lines below correspond to the loss function for the value network
        # This would be equation 5 in the paper
        ##############################################
        self.value_net.optimizer.zero_grad()
        value_loss = 0.5 * F.mse_loss(V_s_t, q_theta_min - sampled_log_pi_t)
        value_loss.backward(retain_graph=True)
        self.value_net.optimizer.step()




        sampled_a_t, sampled_log_pi_t = self.actor.sample_normal(s_t, reparameterize=True)
        sampled_log_pi_t = sampled_log_pi_t.view(-1)
        q_theta_1_new_policy = self.critic_1.forward(s_t, sampled_a_t)
        q_theta_2_new_policy = self.critic_2.forward(s_t, sampled_a_t)
        q_theta_min = T.min(q_theta_1_new_policy, q_theta_2_new_policy)
        q_theta_min = q_theta_min.view(-1)
        
        actor_loss = sampled_log_pi_t - q_theta_min
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*r_t + self.gamma*V_s_t_plus_1
        q1_old_policy = self.critic_1.forward(s_t, replay_buffer_a_t).view(-1)
        q2_old_policy = self.critic_2.forward(s_t, replay_buffer_a_t).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

        actions_array = self.make_numpy(sampled_a_t)
        value_array_ = np.array(V_s_t_plus_1.cpu().detach().numpy())
        value_array = np.array(V_s_t.cpu().detach().numpy())
        q1_new_policy_arr = self.make_numpy(q_theta_1_new_policy)
        q2_new_policy_arr = self.make_numpy(q_theta_2_new_policy)
        critic_value_arr = self.make_numpy(q_theta_min)
        value_loss_arr = self.make_numpy(value_loss)


