# Robo Copilot

An A.I. copilot to aid in robotic arm manipulation.

Progress tracking:

https://docs.google.com/presentation/d/1seex4L58r7VNsQXn5mE32FUOs79kVj9YIQ5Vw28cujA/edit?usp=sharing

#### Contents

- [Description](#description)
  - [System Overview](#system-overview)
  - [Models](#models)
    - [Vision Encoder](#vision-encoder)
    - [Diffusion Policy](#diffusion-policy)
  - [Data](#data)
    - [Proprioceptives](#proprioceptive)
    - [Vision](#vision)
    - [Actions](#actions)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
    - [Configuration](#configuration)
    - [Launch Training](#launch-training)
    - [Resume Training](#resume-training)
- [References](#references)

## Description

We utilize visual and proprioceptive data to enable A.I.-guided robotic arm manipulation.

### System Overview

...diagram here...

### Models

#### Vision Encoder

The vision encoder is a neural network that processes camera inputs to generate vision features used for conditioning. We use a separate instance of vision encoder for each camera view.

We have tried pre-trained encoders, but have generally found that end-to-end trainig out-performs any pre-trained model.

We are using Resnet-18 based encoders, though we have also tried R3M.

#### Diffusion Policy

The diffusion policy generates expert robotic arm action signals from isometric Gaussian noise, conditioned on the vision features from the vision encoders and on the proprioceptive features.

This policy looks at `obs_horizon` frames of past observations and generates `pred_horizon` frames of future actions, of which `action_horizon` frames are executed.

Note that since the diffusion process takes time, one might notice some staggering in the robotic arm movements. Our hope is that when shared autonomy is implemented, actions no longer have to be generated from pure noise, so we don't have to go through as many diffusion iterations.

### Data

#### Proprioceptives

|    time    |       x_pos        |       y_pos        |       z_pos        | roll | pitch | yaw |     gripper_width     |
| :--------: | :----------------: | :----------------: | :----------------: | :--: | :---: | :-: | :-------------------: |
| time in ms | end effector x pos | end effector y pos | end effector z pos | roll | pitch | yaw | dist between grippers |

#### Vision

`cam_0/`: camera views from in front of FRANKA arm.

`cam_1/`: camera views from FRANKA arm gripper's POV.

#### Actions

|    time    |   x_vel    |   y_vel    |   z_vel    | roll | pitch | yaw |    gripper     |
| :--------: | :--------: | :--------: | :--------: | :--: | :---: | :-: | :------------: |
| time in ms | x velocity | y velocity | z velocity | roll | pitch | yaw | gripper toggle |

## Installation

1. Clone a copy of this repo and the r3m repo:

```bash
git clone https://github.com/bmcmahan2016/robo_copilot.git
git clone https://github.com/facebookresearch/r3m.git
```

2. Set up a virtual environment (Conda or Virtualenv) and install `r3m`, `torch`, `torchvision`, `pandas`, `onnx`, `wandb`.

- using `virtualenv` (recommended):

```bash
cd robo_copilot
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision pandas onnx wandb
cd PATH/TO/r3m
pip install -e .
```

- using `conda`:

```bash
conda create -n RoboCopilot
conda activate RoboCopilot
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
conda install conda-forge::wandb
conda install conda-forge::onnx
conda install pandas
cd PATH/TO/r3m
pip install -e .
```

3. Set the `$PYTHONPATH` environment variable to `robo_copilot`. This enables imports of modules in this repo.

```bash
cd robo_copilot
export PYTHONPATH=$(pwd)
```

- note: on servers or on different terminal sessions, you might have to repeat this step.

You are ready to start using **RoboCopilot**!

## Usage
### Data Collection
To collect data with spacemouse:
```
python scripts/spacemouse.py --save_dir YOUR_TARGET_DIR
```
If you just want to practice without saving, no need to pass in the argument.

To replay the collected kinesthetic teaching mode:
```
python scripts/kinesthetic_replay.py --dataset YOUR_HDF5_FILE_PATH
```
Kinesthetic teaching should be collected in Deoxys folder. Please refer to there.

To reset your model to random position with random rotation,
```
reset
```

To reset your model to fixed position,
```
reset --fixed True
```

### Training

#### Configuration

The configuration object is a python `@dataclass` called `BlockPushConfig`. To instanciate one, create a python file in `configs/` like so:

```python
from diffusion_policy.utils import BlockPushConfig

# be sure to name this instance "config"!
config = BlockPushConfig(
    data_dir = "/path/to/dataset",
    save_dir = "/path/to/save_location",
    views = ("cam0_left", "cam1_left", "mini_left"),
    # parameters
    pred_horizon = 16,
    obs_horizon = 2,
    action_horizon = 8,
    #|o|o|                             observations: 2,
    #| |a|a|a|a|a|a|a|a|               actions executed: 8,
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16,
    batch_size = 64,
    # ResNet18 has output dim of 512,
    vision_feature_dim = 512,
    # agent_pos is 2 dimensional,
    lowdim_obs_dim = 4,
    # observation feature has 514 dims in total per step,
    action_dim = 4,
    num_diffusion_iters = 100,
    num_epochs = 2000,
)
```

Note: make sure to call this config object `config`!

#### Launch Training

1. Set up the virtual environment and python path:

```bash
cd robo_copilot
source .venv/bin/activate
export PYTHONPATH=$(pwd)
```

2. Launch training!

```bash
python diffusion_policy/train.py --config path/to/config_file.py
```

This might ask you to log into your Weights&Biases account. Just follow the prompts.

#### Resume Training

If you've trained a policy in the past but want to train it further, you can resume a training by providing model checkpoint through the `--resume` flag. It is also recommended that you provide the wandb run ID associated with this model using the `--wandb_run_id` flag so training metadata can continue to be logged to the same wandb run.

```bash
python diffusion_policy/train.py --config path/to/config_file.py --resume path/to/checkpoint.ckpt --wandb_run_id w&brunid
```

## References

Chi et al 2023 - Diffusion Policy
Showed diffusion-based robotic control policies’ ability to tackle many non-trivial robotic control tasks

Yoneda et al 2023 - Diffusion for Shared Autonomy
Showed diffusion-based policies’ potential for drastically improving the quality of human-generated action signals for many difficult control tasks

Zhang et al 2023 - Neural Signal Operated Intelligent Robots
Showed promise for EEG-based neural signals for robotic control
