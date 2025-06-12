import argparse
import wandb
import gymnasium
from networks import PolicyNet, ValueNet
from reinforce import reinforce
from common import run_episode
from Qnetworks import train_dqn

def parse_args():
    """The argument parser for the main training script."""
    parser = argparse.ArgumentParser(description='A script implementing REINFORCE on the Cartpole environment.')
    parser.add_argument('--project', type=str, default='DLA2025-Cartpole', help='Wandb project to log to.')
    parser.add_argument('--baseline', type=str, default='none', help='Baseline to use (none, std, val_net)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--visualize', action='store_true', help='Visualize final agent')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='CartPole-v1 or LunarLander-v3')
    parser.add_argument('--agent', type=str, default='reinforce', help='Agent to use: reinforce or dqn')
    parser.set_defaults(visualize=False)
    args = parser.parse_args()
    return args


# Main entry point
if __name__ == "__main__":
    # Get command line arguments
    args = parse_args()

    # Initialize wandb with our configuration parameters
    run = wandb.init(
        project=args.project,
        config={
            'agent': args.agent,
            'learning_rate': args.lr,
            'baseline': args.baseline,
            'gamma': args.gamma,
            'num_episodes': args.episodes
        }
    )

    # Select environment
    env_name = "LunarLander-v3"
    args.agent = 'dqn'
    env = gymnasium.make(args.env_name)
    if args.env_name == "LunarLander-v3" and args.agent == 'reinforce':
        args.episodes = 2000  # More episodes for REINFORCE on LunarLander

    # Train the selected agent
    if args.agent == 'reinforce':
        if env_name == "CartPole-v1":
            policy = PolicyNet(env)
            value_net = ValueNet(env)
        else:
            policy = PolicyNet(env, n_hidden=3, width=256) # More complex policy network for LunarLander
            value_net = ValueNet(env, n_hidden=3, width=256) # More complex value network for LunarLander

        reinforce(
            policy, value_net, env, run,
            lr=args.lr,
            baseline=args.baseline,
            num_episodes=args.episodes,
            gamma=args.gamma
        )

        # Visualize REINFORCE policy
        if args.visualize:
            if args.env_name == "CartPole-v1":
                env_render = gymnasium.make('CartPole-v1', render_mode='human')
            else:
                env_render = gymnasium.make('LunarLander-v3', render_mode='human')
            for _ in range(10):
                run_episode(env_render, policy)
            env_render.close()

    elif args.agent == 'dqn':
        train_dqn(
            env,
            run,
            episodes=args.episodes
        )

    # Final cleanup
    env.close()
    run.finish()
