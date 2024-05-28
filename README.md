# DIA Coursework

## The scientific questions we are trying to solve are:

### 1) What is the sensitivity of the agent to variations in the initial positioning of the target?

### 2) How does altering the reward system influence the agent's ability to learn and execute the required sequence of steps to accomplish the task?

## Directory Structure

Every directory has the following files:

- **`brain.py`** - This is where the training happened.
- **`constants.py`** - All constants are defined here.
- **`game.py`** - This implements the environment.
- **`model.py`** - This defines the DQN algorithm.
- **`test_model.py`** - This uses the saved model and runs the environment.

## Running the Project

In order to check the working, 
Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

you could just run the `test_model.py` file after cloning the entire repository!



## Experimental Design

### How experiments are performed:

#### Sensitivity to Initial Positioning of the Target

We conducted experiments by randomly changing the initial positions of the targets in one set and keeping them the same in another. Because the agent behaves non-deterministically, each experiment was run 10 times and the average results were calculated.

#### Altering the Reward System

Three experiments were conducted to assess the impact of different reward structures:

- **Exp1:** +100 for hitting the target; 0 otherwise.
- **Exp2:** +100 for hitting the target, -5 for firing a bullet, -1 for each timestep without hitting the target, 0 otherwise.
- **Exp3:** +100 for hitting the target, -5 for firing a bullet, -1 for each timestep without hitting the target, +1 for being close to the target without hitting it, with no penalty for passing a timestep without hitting the target, 0 otherwise.

## Project Results

The folder `exp_results` contains the results, and the notebook `/dia-sci-question.ipynb` outlines the entire project results and explains the project outcomes.

## References

1. [Patrick Loeber's Snake AI using PyTorch](https://github.com/patrickloeber/snake-ai-pytorch)
2. [Shiva Verma's Orbit Paddle](https://github.com/shivaverma/Orbit/tree/master/Paddle)
3. Laud, Adam, and Gerald DeJong. "The influence of reward on the speed of reinforcement learning: An analysis of shaping." Proceedings of the 20th International Conference on Machine Learning (ICML-03). 2003.
4. Lee, Kimin, et al. "Network randomization: A simple technique for generalization in deep reinforcement learning." arXiv preprint arXiv:1910.05396 (2019).
