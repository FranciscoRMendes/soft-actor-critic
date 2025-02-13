import torch
import gym
import imageio

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model_name = 'Train500000fr_good'
policy_net = torch.load(f'trained_models/{model_name}', weights_only=False, map_location="cpu")
policy_net.eval()
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


