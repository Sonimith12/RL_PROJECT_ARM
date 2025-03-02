distance = np.linalg.norm([x_end - target_x, y_end - target_y])
max_distance = np.sqrt((self.armconfig.SIZE_HUMERUS + self.armconfig.SIZE_RADIUS) ** 2)
normalized_distance = distance / max_distance  # Range [0, 1]

# Exponential reward parameters
k = 10          # Controls steepness (higher = sharper increase near target)
base_reward = 20  # Scales the peak reward

# Exponential proximity bonus (max at distance=0, decays rapidly with distance)
proximity_bonus = base_reward * np.exp(-k * normalized_distance)

# Success bonus and energy penalty
success = 100 if distance < 6 else 0
energy_penalty = 0.001 * np.sum(action)

# Final reward
reward = proximity_bonus + success - energy_penalty