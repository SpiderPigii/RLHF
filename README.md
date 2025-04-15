
# Reinforcement Learning with Human Feedback

The following is our implementation of Reinforcement Learning from Human Feedback (RLHF) based on the CleanRL codebase, using Proximal Policy Optimization (PPO).

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/vwxyzjn/cleanrl)


Our work is based on the [CleanRL](https://github.com/vwxyzjn/cleanrl) codebase, and we followed the paper by [Christiano et al.](https://arxiv.org/abs/1706.03741) to implement Reinforcement Learning from Human Feedback (RLHF).

### Key Features of CleanRL
CleanRL is a deep reinforcement learning library with high-quality, single-file implementations designed for research and scalability. Its key features include:
- 📜 **Single-file implementation**: All algorithm details are contained within standalone files (e.g., `ppo_atari.py` has only 340 lines of code).
- 📊 **Benchmarked implementation**: ([benchmark.cleanrl.dev](https://benchmark.cleanrl.dev)).
- 📈 **Tensorboard logging**: For real-time metrics visualization.
- 🪛 **Local reproducibility**: Full reproducibility via seeding.
- 🎮 **Gameplay videos**: Automatic video recording.
- 💸 **Cloud support**: Easy scaling with Docker and AWS.

For more details, refer to the [CleanRL JMLR paper](https://www.jmlr.org/papers/volume23/21-1342/21-1342.pdf) and [documentation](https://docs.cleanrl.dev/).

</details>

---

## Getting Started

![Sample Video](https://gymnasium.farama.org/_images/half_cheetah.gif)

### Prerequisites
Ensure you have Python 3.7.1 to 3.10 installed.


### Set up a virtual environment with Python 3.9
```bash
python3.9 -m venv env
```
#### To activate on Linux:
```bash
source env/bin/activate
```
#### To activate on Windows: 
```bash
.\env\Scripts\Activate
```
#### Verify Python version
```bash
python --version
```

### Install dependencies
```bash
pip install -r requirements/requirements-mujoco.txt
```

---

### Run in the Terminal
To run the RLHF code directly in the terminal without Human Feedback:
1. Navigate to the main directory.
2. Run one of the following commands:

```bash
python cleanrl/ppo_continuous_action.py --env-id HalfCheetah-v4
```

To monitor training progress: 
1. Open another terminal.
2. Navigate to the `cleanrl` folder:
   ```bash
   cd cleanrl/cleanrl
   ```
3. Run Tensorboard:
   ```bash
   tensorboard --logdir runs
   ```
---

### Notes

- **Non-modular approach**: CleanRL focuses on simplicity over modularity, making it ideal for understanding and prototyping reinforcement learning algorithms.

![](docs/static/o1.png)
![](docs/static/o2.png)
![](docs/static/o3.png)

## Citing CleanRL

If you use CleanRL in your work, please cite our technical [paper](https://www.jmlr.org/papers/v23/21-1342.html)

## Acknowledgement

CleanRL is a community-powered by project and our contributors run experiments on a variety of hardware.

* We thank many contributors for using their own computers to run experiments
* We thank Google's [TPU research cloud](https://sites.research.google/trc/about/) for providing TPU resources.
* We thank [Hugging Face](https://huggingface.co/)'s cluster for providing GPU resources. 
</details>
