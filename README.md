Run one of these commands: 

# Basic training with default parameters
python train_model.py

# With custom learning rate and render
python train_model.py --learning-rate 1e-4 --render

# With GPU acceleration
python train_model.py --use-gpu

# Full custom configuration
python train_model.py \
    --learning-rate 1e-4 \
    --buffer-size 2000000 \
    --batch-size 512 \
    --total-timesteps 2000000 \
    --save-path models/custom_model \
    --log-dir logs/custom_experiment \
    --experiment-name custom_run \
    --use-gpu \
    --render
