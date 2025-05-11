## **Installation**  
```bash
pip install -r requirements.txt
```
### Basic training with default parameters
```bash
python train_model.py
```
### With custom episode-length and total timesteps 
```bash
python train_model.py --use-gpu --episode-length 2500 --total-timesteps 250000
```
### Loading trained model (Consider modifying the name of the model according to the name you save it default: models/sac_arm.zip)
```bash
python load_model.py --model-path models/sac_arm_latest.zip --eval-steps 2500
```
### For more information on other arguments please check train_model.py
