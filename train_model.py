import argparse
import os
import torch
import logging
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from arm_model import ArmReachingEnv2DTheta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
class TrainingLoggerCallback(BaseCallback):
    """Custom callback for logging training progress at each episode."""
    
    def __init__(self, check_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.logger = logging.getLogger(__name__)
        self.episode_rewards = []  # Store cumulative rewards for each episode
        self.episode_lengths = []  # Store lengths of each episode
        self.episode_count = 0     # Track the number of episodes
        self.current_episode_reward = 0  # Track reward for the current episode
        self.current_episode_length = 0  # Track length of the current episode

    def _on_step(self) -> bool:
        # Check if the episode has ended using the "dones" array
        done = self.locals["dones"][0]  # For vectorized environments

        # Accumulate reward and length for the current episode
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        if done:
            # Retrieve termination flags from the info dict
            terminated = self.locals["infos"][0].get("terminated", False)
            truncated = self.locals["infos"][0].get("truncated", False)

            # Store cumulative reward and episode length
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # Log episode information every `check_freq` episodes
            if self.episode_count % self.check_freq == 0:
                env = self.model.env.envs[0]
                training_info = {
                    "episode": self.episode_count,
                    "timesteps": self.model.num_timesteps,
                    "cumulative_reward": self.current_episode_reward,
                    "episode_length": self.current_episode_length,
                    # "avg_muscle_activation": np.mean(env.muscle_activations),
                    "terminated": terminated,
                    # "truncated": truncated,
                }
                log_message = " | ".join(
                    [f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}"
                     for k, v in training_info.items()]
                )
                self.logger.info(f"*** Episode Progress: {log_message} ***")

            # Reset counters for the next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_count += 1

        return True

    def get_episode_rewards(self):
        """Returns the cumulative rewards for all episodes."""
        return self.episode_rewards
def main(args):
    # Initialize the environment
    env = ArmReachingEnv2DTheta(render_mode="human" if args.render else None)
    # env = DummyVecEnv([lambda: env])  # Wrap the environment

    # Set device (CPU or GPU)
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Load or initialize the model
    if args.load_model and os.path.exists(args.save_path):
        logger.info(f"Loading existing model from {args.save_path}...")
        model = SAC.load(args.save_path, env=env, device=device)
    else:
        logger.info("Initializing new model...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            ent_coef=args.ent_coef,  # Let SAC tune entropy automatically
            gamma=args.gamma,
            tau=args.tau,
            device=device,
        )

    logger.info("Training started...")

    # Initialize the callback
    callback = TrainingLoggerCallback(check_freq=1)  # Log every episode

    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback
    )

    logger.info("Training completed.")

    # Save the model if a save path is provided
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        model.save(args.save_path)
        logger.info(f"Model saved to {args.save_path}")

    # Access cumulative rewards for all episodes
    episode_rewards = callback.get_episode_rewards()
    logger.info(f"Cumulative rewards for all episodes: {episode_rewards}")

    # Evaluate the model
    logger.info("Starting evaluation...")
    state, _ = env.reset()
    episode_reward = 0

    for step in range(args.episode_length):
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        
        episode_reward += reward
        logger.info(f"Step: {step}, Reward: {reward:.4f}, Cumulative Reward: {episode_reward:.4f}")

        if terminated or truncated:
            logger.info("Evaluation episode finished!")
            break
    
    logger.info(f"Evaluation completed with total reward: {episode_reward:.4f}")
    env.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a robotic arm using SAC.")
    
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Size of the replay buffer.")
    parser.add_argument("--ent-coef", type=str, default="auto", help="Entropy regularization coefficient.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient for target networks.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total number of timesteps to train.")
    
    # Add episode length as an argument
    parser.add_argument("--episode-length", type=int, default=2500, help="Number of steps per episode.")
    
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available.")
    
    parser.add_argument("--save-path", type=str, default="models/sac_arm", help="Path to save the trained model.")
    parser.add_argument("--load-model", action="store_true", help="Load an existing model before training.")

    parser.add_argument("--render", action="store_true", help="Render environment during training.")

    args = parser.parse_args()
    
    # Run the main function
    main(args)