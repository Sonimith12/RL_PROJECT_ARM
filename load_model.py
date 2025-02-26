# load_model.py
import argparse
import logging
from stable_baselines3 import SAC
from arm_model import ArmReachingEnv2DTheta

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_loaded_model(model_path, eval_steps=500, render=True):
    """Load and evaluate a trained model with rendering."""
    try:

        cumm_reward = 0 
        # Create environment with rendering
        env = ArmReachingEnv2DTheta(render_mode="human" if render else None)
        
        # Load the trained model
        logger.info(f"Loading model from {model_path}...")
        model = SAC.load(model_path, env=env)

        logger.info("Starting evaluation with rendering...")
        state, _ = env.reset()
        
        for step in range(eval_steps):
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            cumm_reward += reward
            logger.info(f"Step: {step}, Reward: {reward:.4f}")
            
            if terminated or truncated:
                logger.info("Episode finished! Resetting environment...")
                state, _ = env.reset()

        env.close()
        logger.info("Evaluation completed.")
        logger.info(f"Total Steps:{step+1}, Total rewawrd: {cumm_reward})") 
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained robotic arm model.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model zip file")
    parser.add_argument("--eval-steps", type=int, default=2500,
                        help="Number of steps for evaluation")
    parser.add_argument("--no-render", action="store_false", dest="render",
                        help="Disable rendering during evaluation")
    
    args = parser.parse_args()
    
    evaluate_loaded_model(
        model_path=args.model_path,
        eval_steps=args.eval_steps,
        render=args.render
    )

# to run: python load_model.py --model-path models/sac_arm.zip --eval-steps 1000