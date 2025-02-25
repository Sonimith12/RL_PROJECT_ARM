## **Installation**  
```bash
pip install -r requirements.txt
```
### Basic training with default parameters
```bash
python train_model.py
```
### With custom learning rate and render
```bash
python train_model.py --learning-rate 1e-4 --render
```
### With GPU acceleration
```bash
python train_model.py --use-gpu
```
### Full custom configuration
```bash
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
```
### Loading trained model 
```bash
python load_model.py --model-path models/sac_arm.zip --eval-steps 1000
```
