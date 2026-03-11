# Supervised_Reinforcement_Learning
This repository is the official implementation for the paper: **"Uncertainty Allocation-based Tube Model Predictive Control for Building Energy Management"**.

Authors: Haoyuan Deng, Yihong Zhou, Thomas Morstyn, Yi Wang

Inspired by the training paradigms in large language models, this paper proposes a Supervised Reinforcement Learning (SRL) framework for learning DER coordination policies. This framework first pre-trains a policy on demonstration data in a supervised-learning fashion, which is then further fine-tuned using RL. Furthermore, we propose a two-step fine-tuning process: offline fine-tuning for enhancing policy performance and online fine-tuning for adapting it to the real-world dynamics.

## Contents

- `main.py` — Main script orchestrating BC pretraining, offline fine-tuning in sim env and online fine-tuning in real env.
- `Env/` — Environment implementation (e.g., `Env/Env.py` and `Env/Battery.py`). Replace or adapt this if you want a different environment. If you use the current environment, please modify the path of the dataset in the current `Env.py' file.
- `test_act/` — Example demonstration data (e.g., `his_Action.npy`). This can be replaced with other expert trajectories.
- `environment.yml` — Conda environment specification for dependencies.

---

## Key features

- Behavior cloning (BC) supervised pretraining via the `imitation` library (`imitation.algorithms.bc`).
- Two-phase fine-tuning: simulated environment (`sim_env`) then real environment (`real_env`).
- Logging (monitor files and TensorBoard) via `Monitor`, `SummaryWriter`, and Stable Baselines3 tensorboard integration.

---

## Quickstart — run the full flow

1. Create environment (recommended):

   ```bash
   conda env create -f environment.yml
   conda activate srlenv  # or whatever name the yml defines
   pip install -e .  # optional for local editable installs
   ```

2. Prepare demonstration data

   - The default demo used by `main.py` is `test_act/his_Action.npy` (scaled in code). You can replace this file with other action trajectories as needed.

3. Configure `main.py` (optional)

   - `model_name`, `output_dir` (default `new_logs/`), and `act_dir` are set near the bottom of `main.py`.
   - To switch environments, replace `Env.EH_Model` or adapt `Env/Env.py` implementation.

4. Run the script:

   ```bash
   python main.py
   ```

   The script performs:
   - `imitation_learning(...)` — loads `his_Action.npy`, runs BC pretraining and returns a `bc_trainer` (policy)
   - `train_ppo_agent(...)` — loads that BC-initialized policy into PPO, fine-tunes in simulation (`sim_env`) and then in the real environment (`real_env`).

5. Check outputs:

   - Models: `new_logs/<run_name>/sim_env/final_model` and `new_logs/<run_name>/real_env/final_model`
   - TensorBoard logs: `logs/episode_reward/` (or the `tensorboard_log` directories under `new_logs/`)

---

## Replacing the environment or demonstration data

- **Environment**: The environment class used is `Env.EH_Model` (in `Env/Env.py`). To use a different environment:
  - Implement a Gym-compatible environment and either replace `Env/Env.py` or import and pass the desired class in `train_ppo_agent`/`imitation_learning`.
  - Ensure your environment follows the same `observation_space` and `action_space` conventions used by PPO and BC.

- **Demonstration data**: By default `main.py` loads `test_act/his_Action.npy`.
  - Your demonstration data should be an array with shape compatible with the action space; the script currently reshapes actions with `.reshape((-1,3))` and rescales each dimension — modify preprocessing as needed.

---

## Dependencies & environment

Major dependencies (see `environment.yml` for exact versions):

- Python 3.8+ (check `environment.yml`) 
- PyTorch
- Stable Baselines3
- gymnasium
- imitation (OpenAI imitation library)
- numpy, matplotlib, tensorboard

**Note:** The PPO algorithm used in this project is provided by **Stable Baselines3**; behavior cloning (BC) is provided by the separate **`imitation`** library (they are designed to interoperate with Stable Baselines3).

---

## Tips & notes

- Use `tensorboard --logdir logs/` to visualize training curves.
- You can bypass simulation and train directly in the real environment by setting appropriate flags in `main.py`.
- To use offline prefill without BC pretraining, construct `prefill_data` and pass it into `train_ppo_agent`.

---

## License

See the included `LICENSE` file.

---
