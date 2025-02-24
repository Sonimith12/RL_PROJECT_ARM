import argparse
import os
import torch
from stable_baselines3 import SAC
from arm_model import ArmReachingEnv2DTheta

def train_model(args):
    env = ArmReachingEnv2DTheta(render_mode="human" if args.render else None)
    
    if args.use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        tau=args.tau,
        device=device
    )

    print("\n🚀 Training Started...\n")
    
    for step in range(1, args.total_timesteps + 1):
        model.learn(total_timesteps=1)

        if step % 200 == 0:
            last_rewards = env.eph.cum_reward_episode if hasattr(env, 'eph') else "N/A"
            
            print(f"🟢 Step {step}/{args.total_timesteps}")
            print(f"   - Cumulative Reward: {last_rewards:.4f}")

            for name, param in model.policy.named_parameters():
                if param.grad is not None:
                    print(f"   - {name} Gradient: {param.grad.norm().item():.6f}")

            print("-" * 40)

    print("\n✅ Training Completed!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train robotic arm SAC model')
    
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--buffer-size', type=int, default=1_000_000,
                        help='Size of the replay buffer')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Mini-batch size for training')
    parser.add_argument('--ent-coef', type=str, default='auto',
                        help='Entropy regularization coefficient')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient for target networks')
    parser.add_argument('--total-timesteps', type=int, default=1_000_000,
                        help='Total number of timesteps to train')
    
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')

    args = parser.parse_args()
    
    train_model(args)
