# Demonstrations, training, running model

## Collecting demonstrations

### Run
```
roslaunch /home/palpatine/catkin_ws/src/panda_zed/zed_wrapper/launch/multi_camera.launch
```

Alternatively, `runpanda` is a bash function defined in `~/.bashrc` that runs the above command.

When collecting multiple demonstrations consecutively, consider using `runpandaforever` which loops the above command.

### Configuration

In `~/catkin_ws/src/panda_zed/zed_wrapper/launch/multi_camera.launch`
```xml
<launch>
    <arg name="data_path" default="/home/palpatine/data/panda/[data_dir]"/>
    <!-- Launch the zedm.launch with custom arguments -->
    ...
</launch>
```

Replace `[data_dir]` with desired directory name where demonstration data will be placed under (e.g. `/home/palpatine/data/panda/var_src_fix_des_3`). No need to preemptively create the directory.

### Adding noise
Uncomment lines 384 through 394 in `~/catkin_ws/src/cpp_panda/tests/move_joystick.cpp`. Robot motor seems to not like this a whole lot (makes scratchy sounds). Adjust parameters in global variables `NOISE_FACTOR` and `NOISE_DECAY`.

## Running a trained model

### Run
```
roslaunch /home/palpatine/catkin_ws/src/panda_zed/zed_wrapper/launch/multi_camera_control.launch
```

No bash shortcut here yet :>, although a relative path can be used instead of an absolute path, so may be easier to `cd` into the directory and run `roslaunch multi_camera_control.launch`

### Configuration

In `~/catkin_ws/src/panda_zed/zed_wrapper/launch/multi_camera_control.launch`
```xml
<launch>
    <arg name="data_path" default="/home/palpatine/data/panda/model_run"/>
    <arg name="model_path" default="/home/palpatine/git/robo_copilot/runs/r3m/[output_dir]/ckpts/model[##].pth"/>
    <!-- Launch the zedm.launch with custom arguments -->
    ...
</launch>
```

Replace `[output_dir]` with the [output directory that contains desired model](#configuration-2) and `[##]` with the desired model number to run (e.g. `/home/palpatine/git/robo_copilot/runs/r3m/var_src_fix_des_3/ckpts/model100.pth`).

## Launch training

`cd` into `~/git/robo_copilot`.

### Run

```
python3 scripts/run_experiment.py configs/r3m_policy.yaml
```

### Configuration

In `configs/r3m_policy.yaml`
```yaml
train_dir: /home/palpatine/data/panda/var_src_fix_des_3
val_dir: /home/palpatine/data/panda/var_src_fix_des_3
proprioceptive: 'proprioceptive.csv'
actions: 'action.csv'
img_prefix: 'Image'
camera_folder: 'cam0_left'
view_folder: ''
device: 'cuda:0'
output_dir: 'runs/r3m/var_src_fix_des_3'
save_freq: 1
drop_images: False
model_name: R3M_MLP
hyperparams: 'experiments/hyperparameters.yaml'
num_workers: 8
last_epoch: 0
```

Set `train_dir` and `val_dir` to corresponding `[data_dir]` from [configuration on trained model](#configuration).

Set `output_dir` to desired directory name to store model checkpoints.

*New parameter*: If attempting to continue training from a previous checkpoint, set `output_dir` and `last_epoch` to model and epoch to continue training from. Training loop will begin on `last_epoch + 1`. Make sure that `train_dir` and `val_dir` are correct. Set `last_epoch` to `0` when training new model from scratch

In `experiments/hyperparameters.yaml`
```yaml
learning_rate: 0.001
num_epochs: 200
steps_per_epoch: 1000
input_size: 2052
hidden_size: 256
output_size: 4
batch_size: 32
```

Adjust `learning_rate` as desired. When `last_epoch` is set to a value greater than zero, the training loop will train epochs `last_epoch + 1` to `last_epoch + num_epochs`, inclusive.

# Other notes

- I recently added `gripper_state` (-1 or 1) and `gripper` (0 or 1) to proprioceptive and action data respectively. This corresponds to a number of changes in the code that may be incompatible with previously collected demonstrations. List of things that may need to be changed if running without gripper data:
    - `output_size` in `~/git/robo_copilot/experiments/hyperparameters.yaml` set from 4 to 3
    - `input_size` in `~/git/robo_copilot/experiments/hyperparameters.yaml` set from 2052 to 2051
    - same two parameters above in `~/catkin_ws/src/robot_action/src/robot_action/configs/hyperparameters.yaml`
    - in `~/git/robo_copilot/models/trainer.py` comment line 76 and uncomment line 77
    - in `~/catkin_ws/src/robot_action/src/generate_actions.py` comment line 82, uncomment line 83, comment line 89
    - may be other problems but hopefully not !
- Also recently changed name of `proprioceptive.csv` and `action.csv` from `position.csv` and `input.csv` respectively. Change `r3m_config.yaml` configurations if running older demonstration data
- If when collecting demonstrations the arm crashes, just reset, delete demonstration (find corresponding directory and `rm -r 2024-08-...`), and start demonstrations again.
- It looks like `self._r3m.eval()` didn't disable training for R3M. Lines 22-23 in `robo_copilot/models/model.py`.

## Project structure

Four directories of note:
1. `~/catkin_ws/src/cpp_panda/`
    - controls arm motion with joystick or trained model
2. `~/catkin_ws/src/panda_zed/zed_wrapper/`
    - data collection
3. `~/git/robo_copilot`
    - model + train
4. `~/catkin-ws/src/robot_action/`
    - symlinks `model.py` from `~/git/robo_copilot`
    - runs trained model

## Notable collected datasets

| Name | Task | Blue block position | Yellow block position | Noise? | Count | [Gripper data?](#other-notes) |
| --- | --- | --- | --- | --- | --- | --- |
| `fix_src_fix_des_2` | block push (blue to yellow) | fixed | fixed | no | 30 | no |
| `fix_src_fix_des_noise` | block push (blue to yellow) | fixed (to side) | fixed (to side) | yes | 20 | no |
| `var_find_1` | move end effector to yellow | n/a | variable | no | 60 | no |
| `var_src_fix_des_3` | block push (blue to yellow) | variable | fixed | no | 50 | yes |
| `pick_place_1` | pick up blue, place on top yellow | variable | variable | no | 21 | yes |
| `pick_place_2` | pick up blue, place on top yellow | variable | fixed | no | 31 | yes |
| `big_block_push` | push big block forward (variable start) | - | - | no | 30 | yes |

## Notable trained models

| Name | Dataset | R3M trained? | Last epoch | Results | LR |
| --- | --- | --- | --- | --- | --- |
| `batchnorm_off_imgs` | `fix_src_fix_des_2` | yes | 153 | Great performance | 0.001 |
| `batchnorm_off_noise` | `fix_src_fix_des_noise` | yes | 200 | Good performance, but misses block sometimes | 0.001 |
| `batchnorm_off_var_find` | `var_find_1` | yes | 200 | Decent performance, misses target sometimes but moves in general direction | 0.001 |
| `r3m_fixed_fix_src_fix_des` | `fix_src_fix_des_2` | no | 200 | Poor, loss significantly larger than `batchnorm_off_imgs` | 0.01 |
| `r3m_fixed_fix_src_fix_des_noise` | `fix_src_fix_des_noise` | no | 200 | ^ | 0.01 |
| `r3m_fixed_var_find` | `var_find_1` | no | 200 | ^ | 0.01 |
| `r3m_fixed_fix_src_fix_des_2` | `fix_src_fix_des_2` | no | 200 | Poor, loss significantly larger than `batchnorm_off_imgs` | 0.001 (trained above models with 0.01 LR on accident, 0.001 still does poor but gets slightly lower loss) |
| `r3m_train_big_block_push` | `big_block_push` | yes | 200 | Good performance to right of initial position, struggles where block is obstructed by arm in camera view | 0.001 |