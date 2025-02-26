import gymnasium as gym
from typing import Optional, Union
from gymnasium import spaces
import numpy as np
import math
import random
from Utils import *
from collections import namedtuple
from Renderer import ArmRenderer
from scipy.interpolate import CubicSpline

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

SEED = 19930515
MAX_EPISODE_STEPS = 2500
FIXED_TARGET = True

Armconfig = namedtuple('Armconfig', ['SIZE_HUMERUS', 'WIDTH_HUMERUS', 'SIZE_RADIUS','WIDTH_RADIUS'])

class ArmReachingEnv2DTheta(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(
        self, render_mode: Optional[str] = None
    ):
        super().reset(seed=SEED)
        # Biomechanical parameters
        self.F_max = 10             # Maximum muscle force 
        self.r_shoulder = 0.05      # Shoulder moment arm
        self.r_elbow = 0.03         # Elbow moment arm 
        self.I_shoulder = 0.1       # Shoulder moment of inertia 
        self.I_elbow = 0.05         # Elbow moment of inertia 
        self.b = 0.1                # Viscous damping coefficient 
        self.dt = 0.05              # Time step 
        
        high_action_range = np.array(
            [
                1,  # shoulder_flexor
                1,  # shoulder_extensor
                1,  # elbow_flexor
                1,  # elbow_extensor
                1,  # biarticular_1
                1,  # biarticular_2
            ],
            dtype=np.float32,
        )

        high_obs = np.array(
            [
                np.inf,  
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(0, high_action_range, dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        self.render_mode = render_mode
        self.state: np.ndarray | None = None
        self.steps_beyond_terminated = None

        self.omega_shoulder = 0  # Angular velocity in rad/s
        self.omega_elbow = 0     # Angular velocity in rad/s

        self.armconfig = Armconfig(SIZE_HUMERUS=200, WIDTH_HUMERUS=20, SIZE_RADIUS=300, WIDTH_RADIUS=10)
        self.armrenderer = None

        self.target_angle_deg = None
        self.theta_1_c = 70  # Initial shoulder angle, we can initialize it randomly: random.uniform(0, 180)
        self.theta_2_c = 0   # Initial elbow angle, same also here: random.uniform(0, 150)

        # Muscle activations
        self.muscle_activations = np.zeros(6, dtype=np.float32)
        self.muscle_names = ['shoulder_flexor', 'shoulder_extensor', 
                            'elbow_flexor', 'elbow_extensor', 
                            'biarticular_1', 'biarticular_2']

        # Reward parameters
        self.epsilon_target = 100
        self.target_radius = 30
        self.eph = None

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        terminated = (self.eph.nb_step_done >= MAX_EPISODE_STEPS-1)
        truncated = False
        self.eph.past_theta_1_values.append(self.theta_1_c)
        self.eph.past_theta_2_values.append(self.theta_2_c)

        # Update muscle activations
        self.muscle_activations = np.clip(action, 0, 1)

        # Calculate joint torques
        shoulder_flexor = self.muscle_activations[0]
        shoulder_extensor = self.muscle_activations[1]
        elbow_flexor = self.muscle_activations[2]
        elbow_extensor = self.muscle_activations[3]

        tau_shoulder = (shoulder_flexor - shoulder_extensor) * self.F_max * self.r_shoulder
        tau_elbow = (elbow_flexor - elbow_extensor) * self.F_max * self.r_elbow

        alpha_shoulder = (tau_shoulder - self.b * self.omega_shoulder) / self.I_shoulder
        alpha_elbow = (tau_elbow - self.b * self.omega_elbow) / self.I_elbow

        self.omega_shoulder += alpha_shoulder * self.dt
        self.omega_elbow += alpha_elbow * self.dt

        delta_theta_1 = math.degrees(self.omega_shoulder * self.dt)
        delta_theta_2 = math.degrees(self.omega_elbow * self.dt)
        self.theta_1_c += delta_theta_1
        self.theta_2_c += delta_theta_2

        self.theta_1_c = np.minimum(np.maximum(self.theta_1_c, 0), 180)
        self.theta_2_c = np.minimum(np.maximum(self.theta_2_c, 0), 150)


        theta_1_rad = np.radians(self.theta_1_c)
        theta_2_rad = np.radians(self.theta_2_c)
        x_end = (
            self.armconfig.SIZE_HUMERUS * np.cos(theta_1_rad) +
            self.armconfig.SIZE_RADIUS * np.cos(theta_1_rad + theta_2_rad)
        )
        y_end = (
            self.armconfig.SIZE_HUMERUS * np.sin(theta_1_rad) +
            self.armconfig.SIZE_RADIUS * np.sin(theta_1_rad + theta_2_rad)
        )

        # Get target position
        # target_x, target_y = self.target_cartesian(self.eph.nb_step_done, MAX_EPISODE_STEPS - 1)
        # self.eph.nb_step_done += 1 
        target_x, target_y = self.target_cartesian[self.eph.nb_step_done]
        
        # Update state
        self.state = np.array(
            [
                self.theta_1_c,
                self.theta_2_c,
                self.omega_shoulder,
                self.omega_elbow,
                x_end - target_x, 
                y_end - target_y
            ],
            dtype=np.float32,
        )

        distance = np.linalg.norm([x_end - target_x, y_end - target_y])
        # Example improved reward function
        
        # Define maximum possible distance
        max_distance = np.sqrt((self.armconfig.SIZE_HUMERUS + self.armconfig.SIZE_RADIUS) ** 2)

        # Normalize distance
        normalized_distance = distance / max_distance

        # Define success bonus
        # success = 100 if distance < 5 else 0

        # Define energy penalty
        # energy_penalty = 0.0001 * np.sum(action)

        # Combine into reward
        # reward = -distance + success
        # distance = np.linalg.norm([x_end-target_x, y_end-target_y])

        # success = 100 if distance < 5 else 0

        # energy_penalty = 0.001 * np.sum(action)
        success = 100 if distance < 6 else 0
        proximity_bonus = (1 - normalized_distance) * 10  # Scale bonus
        energy_penalty = 0.001 * np.sum(action)
        reward = proximity_bonus + success - energy_penalty
        # # reward = -distance + success - energy_penalty
        # reward = -distance+success

        # Update episode history
        self.eph.current_reward = reward
        self.eph.cum_reward_episode += reward
        self.eph.past_action = action

        # Render if needed
        if self.render_mode == "human":
            self.render()

        # Return step results
        
        self.eph.nb_step_done += 1
        info = {"terminated": terminated, "truncated": truncated}
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        
        low, high = 0, 180
        theta_init = [70, 0]  # Initial shoulder and elbow angles
        self.theta_1_c = theta_init[0]
        self.theta_2_c = theta_init[1]
        
        self.steps_beyond_terminated = None

        if FIXED_TARGET:
            # Define intervals for target radius changes (start_step, end_step, radius_scale)
            intervals = [
                (0, 500, 1),            # First 100 steps: full radius
                (500, 1200, 0.5),        # Next 200 steps: half radius
                (1200, 1800, 0.25),       # Next 300 steps: quarter radius
                (1800, MAX_EPISODE_STEPS, 0.125)  # Remaining steps: eighth radius
            ]
            # intervals = [
            #     (0, 200, 1),            # First 100 steps: full radius
            #     (200, 400, 0.9),        # Next 200 steps: half radius
            #     (400, 600, 0.8),       # Next 300 steps: quarter radius
            #     (600, 800, 0.7),       # Next 300 steps: quarter radius
            #     (800, 1000, 0.75),       # Next 300 steps: quarter radius
            #     (1000, 1200, 0.6),
            #     (1200, 1400, 0.65),
            #     (1400, 1600, 0.5),
            #     (1800, MAX_EPISODE_STEPS, 0.125) 
            # ]
            self.target_cartesian = []
            self.target_angle_deg = []
            
            for start, end, scale in intervals:
                radius = self.armconfig.SIZE_HUMERUS + (self.armconfig.SIZE_RADIUS * scale)
                target = sample_point_on_circle(radius)
                target = np.around(target, 2).tolist()
                target_angle_rad = math.atan2(target[1], target[0])
                target_angle = math.degrees(target_angle_rad)
                # Clamp the target angle between 0 and 180
                target_angle = np.clip(target_angle, low, high)
                num_steps = end - start
                self.target_cartesian.extend([target] * num_steps)
                self.target_angle_deg.extend([target_angle] * num_steps)

            self.target_cartesian = self.target_cartesian[:MAX_EPISODE_STEPS]
            self.target_angle_deg = self.target_angle_deg[:MAX_EPISODE_STEPS]
        else:
            sampling_indices = np.linspace(0, MAX_EPISODE_STEPS - 1, 10, dtype=int)
            targets_cartesian = [sample_point_on_circle(self.armconfig.SIZE_HUMERUS + self.armconfig.SIZE_RADIUS) for _ in sampling_indices]
            target_angle_deg = []

            for t in targets_cartesian:
                target_angle_rad = math.atan2(t[1], t[0])
                target_angle_deg.append(math.degrees(target_angle_rad))

            spline = CubicSpline(sampling_indices, target_angle_deg)
            x_interpolated = np.linspace(0, MAX_EPISODE_STEPS, MAX_EPISODE_STEPS)
            self.target_angle_deg = spline(x_interpolated)

            # Clamp the target angle between 0 and 180
            self.target_angle_deg = np.clip(self.target_angle_deg, low, high)
            self.target_angle_deg = [round(t, 2) for t in self.target_angle_deg]

            self.target_cartesian = []
            for t in self.target_angle_deg:
                radius = self.armconfig.SIZE_HUMERUS + self.armconfig.SIZE_RADIUS
                x = radius * np.cos(degrees_to_radians(t))
                y = radius * np.sin(degrees_to_radians(t))
                self.target_cartesian.append(np.around([x, y], 2).tolist())


        self.eph = EpisodeHistory(MAX_EPISODE_STEPS,
                                  self.epsilon_target, 
                                  self.target_radius, 
                                  self.target_angle_deg, 
                                  self.target_cartesian)
        self.eph.nb_step_done = 0
        theta_1_rad = np.radians(self.theta_1_c)
        theta_2_rad = np.radians(self.theta_2_c)
        x_end = (
            self.armconfig.SIZE_HUMERUS * np.cos(theta_1_rad) +
            self.armconfig.SIZE_RADIUS * np.cos(theta_1_rad + theta_2_rad)
        )
        y_end = (
            self.armconfig.SIZE_HUMERUS * np.sin(theta_1_rad) +
            self.armconfig.SIZE_RADIUS * np.sin(theta_1_rad + theta_2_rad)
        )

        # Get initial target position
        target_x, target_y = self.target_cartesian[0]

        # Define the initial state
        self.state = np.array(
            [
                self.theta_1_c,          # Shoulder angle
                self.theta_2_c,          # Elbow angle
                self.omega_shoulder,     # Shoulder angular velocity
                self.omega_elbow,        # Elbow angular velocity
                x_end - target_x,        # X error (end effector - target)
                y_end - target_y         # Y error (end effector - target)
            ],
            dtype=np.float32,
        )

        if self.render_mode == "human":
            self.armrenderer = ArmRenderer(self.metadata, self.armconfig, self.eph)
            self.render()

        return np.array(self.state, dtype=np.float32), {}
    
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        
        self.armrenderer.render()

    def close(self):
        if self.render_mode == 'human':
            self.armrenderer.close()
def main():
    # Initialize the environment
    env = ArmReachingEnv2DTheta(render_mode="human")
    # env = ArmReachingEnv2DTheta(render_mode=None)
    # check_env(env)
    state, _ = env.reset()
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        ent_coef='auto',  # Let SAC tune entropy automatically
        gamma=0.99,
        tau=0.005,        # Soft update coefficient
    )

    model.learn(total_timesteps=1_000_000)
    state, _ = env.reset()

    num_episodes = 25  # Set the number of episodes
    for episode in range(num_episodes):
        state, _ = env.reset()
        for step in range(MAX_EPISODE_STEPS):
            action = env.action_space.sample()  # Take action
            state, reward, terminated, truncated, info = env.step(action)
            print(f"Episode {episode + 1}, Step {step + 1}, Reward: {reward}")

            if terminated or truncated:
                print(f"Episode {episode + 1} finished after {step + 1} steps!")
                break  # End episode if needed

    env.close()
if __name__ == "__main__":
    main()
