# Supervised_Reinforcement_Learning
This repository is the official implementation for the paper: **"Uncertainty Allocation-based Tube Model Predictive Control for Building Energy Management"**.

Authors: Haoyuan Deng, Yihong Zhou, Thomas Morstyn, Yi Wang

Inspired by the training paradigms in large language models, this paper proposes a Supervised Reinforcement Learning (SRL) framework for learning DER coordination policies. This framework first pre-trains a policy on demonstration data in a supervised-learning fashion, which is then further fine-tuned using RL. Furthermore, we propose a two-step fine-tuning process: offline fine-tuning for enhancing policy performance and online fine-tuning for adapting it to the real-world dynamics.

# Environment
Python version: 3.9.21

The must-have packages can be installed by running
```bash
conda env create -f environment.yml
```

