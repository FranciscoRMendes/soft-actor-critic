import torch
import gym
import imageio
from ll_utils.sac import SoftActorCritic
from ll_utils.utils import NormalizedActions

env = NormalizedActions(gym.make("LunarLanderContinuous-v2"))
# env = gym.make("LunarLanderContinuous-v2")

# test environment with an action and a step
action = env.action_space.sample()
state = env.reset()
next_state, reward, done, _ = env.step(action)

# store information about action and state dimensions
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model_name = 'Train_500000_2025-02-13-01-39-25'
file_name = f"./trained_models/{model_name}"
sac = SoftActorCritic.from_file(file_name, state_dim=state_dim, action_dim=action_dim, max_action=1.0, device="cpu")
policy_net = sac.pi_phi
sac.pi_phi.eval()
# policy_net = torch.load(f'trained_models/{model_name}', weights_only=False, map_location="cpu")
# policy_net.eval()
from ll_utils.utils import NormalizedActions

env = NormalizedActions(gym.make("LunarLanderContinuous-v2"))
frames = []  # List to store frames

state = env.reset()
done = False
while not done:
    state = torch.FloatTensor(state).unsqueeze(0)
    action = policy_net.get_action(state).detach()
    next_state, reward, done, _ = env.step(action.numpy())
    state = next_state
    # env.render()
    # Render and store frame
    frame = env.render(mode='rgb_array')
    frames.append(frame)

# save as gif
# Save frames as GIF
# Save frames as GIF (ensuring duration is under 6 seconds)
# imageio.mimsave('gameplay.gif', frames, duration=min(100, 5000 // len(frames)))
imageio.mimsave(f'game_play/videos/{model_name}_gameplay.mp4', frames, fps=min(len(frames) // 6, 30), codec='libx264')

# Close environment
env.close()


