import pybullet_envs
import gym
import numpy as np
from sac_utils.SoftActorCritic import SoftActorCritic
from sac_utils.utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v0')
    soft_actor_critic = SoftActorCritic(input_dims=env.observation_space.shape, env=env,
                                        n_actions=env.action_space.shape[0])
    n_games = 250
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'inverted_pendulum.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    R = [] # total rewards
    load_checkpoint = False

    if load_checkpoint:
        soft_actor_critic.load_models()
        env.render(mode='human')

    for i in range(n_games):
        s_t = env.reset()
        done = False
        R_T = 0
        while not done:
            action = soft_actor_critic.choose_action(s_t)
            s_t_plus_1, r_t, done, info = env.step(action)
            R_T += r_t
            soft_actor_critic.remember(s_t, action, r_t, s_t_plus_1, done)
            if not load_checkpoint:
                soft_actor_critic.learn()
            s_t = s_t_plus_1
        R.append(R_T)
        avg_score = np.mean(R[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                soft_actor_critic.save_models()

        print('episode ', i, 'score %.1f' % R_T, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, R, figure_file)

