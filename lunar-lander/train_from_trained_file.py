# in this script we will load a trained sac agent and train it for longer
import datetime

import torch
from ll_utils.sac import SoftActorCritic
from ll_utils.utils import NormalizedActions, plot
import gym
trained_model_dir = 'trained_models'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# import the environment
env = NormalizedActions(gym.make("LunarLanderContinuous-v2"))
# env = gym.make("LunarLanderContinuous-v2")

# test environment with an action and a step
action = env.action_space.sample()
state = env.reset()
next_state, reward, done, _ = env.step(action)

# store information about action and state dimensions
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
file_name = "./trained_models/Train_3000_2025-02-12-18-19-11"

n_episodes = 5000
# max_frames = 500000
max_frames = 50000
max_steps = 500
frame_idx = 0
rewards = []
batch_size = 128
start_episode = 0

sac = SoftActorCritic.from_file(file_name, state_dim=state_dim, action_dim=action_dim, max_action=1.0, device="cpu")

while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0

    # The inner loop is for the individual steps within an episode
    # here we sample an action from the policy network or random
    # and record state,action,reward,next state and done in the replay buffer
    for step in range(max_steps):
        if frame_idx > 1500:
            # action = policy_net.get_action(state).detach()
            action = sac.choose_action(state)
            # next_state, reward, done, _ = env.step(action.numpy())
            next_state, reward, done, _ = env.step(action.cpu()[0].numpy())
        else:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

        # if start_episode%10 == 0:
        #     env.render(mode = 'rgb_array')

        sac.replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        # we do network updates in each run of the inner loop after recording the buffer
        if len(sac.replay_buffer) > batch_size:
            sac.update(batch_size)

        if done:
            break
    state = {'epoch': frame_idx + 1, 'state_dict': sac.policy_net.state_dict(),
             'optimizer': sac.policy_optimizer.state_dict(), 'rewards': rewards}
    # torch.save(state, 'resumeTrain500000fr')
    start_episode += 1

    print("\r frame {} reward: {}".format(frame_idx, episode_reward))
    rewards.append(episode_reward)

plot(frame_idx, rewards)
todays_date_and_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# torch.save(sac.policy_net, f'Train_{max_frames}')
sac_file = f'Train_{max_frames}_{todays_date_and_time}'
sac_file = f'{trained_model_dir}/{sac_file}'
sac.save(sac_file)