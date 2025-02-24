# train_model.py
import argparse
import os
import torch
from stable_baselines3 import SAC
from arm_model import ArmReachingEnv2DTheta  # Assuming your env is in arm_model.py

def train_model(args):
    # Create environment
    env = ArmReachingEnv2DTheta(render_mode="human" if args.render else None)
    
    # Check for GPU availability
    if args.use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")

    # Create model with command-line arguments
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
        tensorboard_log=args.log_dir if args.log_dir else None
    )

    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name=args.experiment_name,
        log_interval=4
    )

    # Save the model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        model.save(args.save_path)
        print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train robotic arm SAC model')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--buffer-size', type=int, default=1_000_000,
                        help='Size of the replay buffer')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Mini-batch size for training')
    parser.add_argument('--ent-coef', type=float, default='auto',
                        help='Entropy regularization coefficient')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient for target networks')
    parser.add_argument('--total-timesteps', type=int, default=1_000_000,
                        help='Total number of timesteps to train')
    
    # Experiment management
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--save-path', type=str, default='models/sac_arm',
                        help='Path to save trained model')
    parser.add_argument('--log-dir', type=str, default='logs/',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--experiment-name', type=str, default='sac_arm',
                        help='Name for TensorBoard experiment')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')

    args = parser.parse_args()
    
    # Create log directory if specified
    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    
    # Start training
    train_model(args)