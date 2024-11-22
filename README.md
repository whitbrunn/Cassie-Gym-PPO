# RL for Stability Control of Bipedal Robots - Group 43_ME5418
## 1 what we have

- An OpenAI Gym-based Cassie simulation environment*[1]
- LSTM network architectures of policy and critic function
- A PPO-based Learning agent*[2]
- An Intuitive Tensorboard-based training process demonstration
- A straightforward and well-looking video interface to display the trained effect

## 2 How to use

2.1 Installation

```
python==3.8
pytorch==1.8.2
# CUDA 11.1 (Linux)
# NOTE: 'nvidia' channel is required for cudatoolkit 11.1 <br> <b>NOTE:</b> Pytorch LTS version 1.8.2 is only supported for Python <= 3.8.
# conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
gym==0.21.0
pandas
numpy
```


2.2 Usage

First, enter the main directory, and run `train.py` with your customer parameters.

Second, run the following command to check the training process. Especially, please scroll down to the bottom to check the **Test Return** and **Train Return**.

```
tensorboard --logdir=./trained_model/ppo/me5418-Cassie-v0 --port=6009
```


Third, run `eval.py` directly, and a video will show out.

```
python eval.py model_path=./trained_models/ppo/me5418-Cassie-v0/xxx-seed42/actor.pt
```


## 3 One more thing

The authors would like to express his heartfelt thanks to Prof. Guillaume Adrien Sartoretti, for his invaluable guidance, and all the TA He Chenyang, Li Hongyi and Li Peizhuo, for thier selfless and helpful advice.




---
*[1] Adapted from https://github.com/osudrl/cassie-mujoco-sim, https://github.com/lbermillo/cassie-curriculum-learning/ and https://github.com/XHN-1/Cassie_mujoco_RL.*
*[2] Adapted from https://github.com/nikhilbarhate99/PPO-PyTorch.*

# Cassie-Gym-PPO
