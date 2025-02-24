import argparse
import os
import torch
import logging
from stable_baselines3 import SAC
from arm_model import ArmReachingEnv2DTheta
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import logging


# training info
class TrainingLoggerCallback(BaseCallback):
    """Custom callback for logging training progress at each step."""
    
    def __init__(self, check_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq  # Log every `check_freq` steps
        self.logger = logging.getLogger(__name__)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Access training metrics
            training_info = {
                "timesteps": self.model.num_timesteps
                # "exploration_rate": self.model.exploration_rate
            }
            
            # Access environment-specific metrics
            if hasattr(self.model.env, 'envs'):
                env = self.model.env.envs[0]
                if hasattr(env, 'eph'):
                    training_info.update({
                        "current_reward": env.eph.current_reward,
                        "cumulative_reward": env.eph.cum_reward_episode,
                        "avg_muscle_activation": np.mean(env.muscle_activations)
                    })

            # Format and log the training information
            log_message = " \n ".join(
                [f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}"
                 for k, v in training_info.items()]
            )
            self.logger.info(f"Step Progress: {log_message}")

            # Log gradient norms
            # gradient_norms = {}
            # for name, param in self.model.policy.named_parameters():
            #     if param.grad is not None:
            #         gradient_norms[f"grad_{name}"] = param.grad.norm().item()
            
            # if gradient_norms:  # Only log if gradients are available
            #     grad_message = " | ".join(
            #         [f"{k}: {v:.6f}" for k, v in gradient_norms.items()]
            #     )
            #     self.logger.info(f"Gradient Norms: {grad_message}")
            
        return True
    
def train_model(args):
    """Trains the SAC model and evaluates it after training."""
    
    env = ArmReachingEnv2DTheta(render_mode="human" if args.render else None)
    
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    if args.load_model and os.path.exists(args.save_path):
        logger.info(f"Loading existing model from {args.save_path}...")
        model = SAC.load(args.save_path, env=env, device=device)
    else:
        logger.info("Initializing new model...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,  # Enable internal logging
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            tau=args.tau,
            device=device,
            # tensorboard_log=args.log_dir if args.log_dir else None
        )

    logger.info("Training started...")

    model.learn(
        total_timesteps=args.total_timesteps,
        # tb_log_name=args.experiment_name,
        # log_interval=4,
        callback=TrainingLoggerCallback(check_freq=1000)
    )

    logger.info("Training completed.")

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        model.save(args.save_path)
        logger.info(f"Model saved to {args.save_path}")

    evaluate_model(model, env, args)

def evaluate_model(model, env, args):
    """Evaluates the trained SAC model by running a test episode."""
    
    logger.info("Starting evaluation...")
    state, _ = env.reset()

    for step in range(args.eval_steps):
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        
        logger.info(f"Step: {step}, Reward: {reward:.4f}, Terminated: {terminated}")

        if terminated or truncated:
            logger.info("Evaluation episode finished!")
            break
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a robotic arm using SAC.")
    
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="Size of the replay buffer.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for training.")
    parser.add_argument("--ent-coef", type=str, default="auto", help="Entropy regularization coefficient.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.05, help="Soft update coefficient for target networks.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total number of timesteps to train.")
    
    parser.add_argument("--eval-steps", type=int, default=500, help="Number of steps for evaluation.")
    
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available.")
    
    parser.add_argument("--save-path", type=str, default="models/sac_arm", help="Path to save the trained model.")
    parser.add_argument("--load-model", action="store_true", help="Load an existing model before training.")

    # parser.add_argument("--log-dir", type=str, default="logs/", help="Directory for TensorBoard logs.")
    parser.add_argument("--experiment-name", type=str, default="sac_arm", help="Name for TensorBoard experiment.")
    parser.add_argument("--render", action="store_true", help="Render environment during training.")

    args = parser.parse_args()
    
    # if args.log_dir and not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir, exist_ok=True)
    
    train_model(args)
