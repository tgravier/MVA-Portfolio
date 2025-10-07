# Course: Reinforcement Learning

### Course Context

- **Professor:** Emmanuel Rachelson (ISAE-SUPAERO)  
- **Program:** Master MVA – Mathematics, Vision, Learning  
- **ECTS:** 5  
- **Final Grade:** 17.50 / 20  
- **Project Grade:** 7 / 9  
- **Year:** 2024 – 2025  

This course provides a rigorous introduction to **reinforcement learning** (RL): modeling decision-making problems, solving Markov Decision Processes, and extending to modern deep RL approaches (DQN, policy gradients, actor-critic, exploration strategies).

---

### Core Topics Covered

- Markov Decision Processes (MDPs), policies, value functions  
- Bellman equations, dynamic programming (value/policy iteration)  
- Model-free learning: Monte Carlo, Temporal Difference (TD) methods  
- Q-Learning, Deep Q-Networks (DQN) and extensions  
- Policy gradient methods and actor-critic algorithms  
- Exploration vs. exploitation: UCB, Thompson Sampling, ε-greedy  
- Continuous actions, function approximation, stability in RL  
- Advanced algorithms: DDPG, SAC, prioritized replay, double networks  

---

### Final Project — *Project Litr0ck*

The repository you linked (**https://github.com/RL-MVA-2024-2025/assignment-Litr0ck**) is the official project assignment and the solution for the final project component of the course.

#### **Project Subject**

- The project revolves around designing, implementing, and evaluating RL agents on a chosen environment (often a control or game environment).  
- The task involves applying RL algorithms from the course (e.g. DQN, policy gradient) and possibly extensions (improvements, stabilizations, hyperparameter tuning) to obtain strong performance.  
- You are expected to analyze convergence, stability, and compare variants of algorithms.  
- The solution repository contains notebooks, training scripts, results, and documentation of your approach.

#### **My Contributions & Experiments**

- I used the GitHub **assignment-Litr0ck** repository as the base of my project, implementing my own solution variant based on that baseline.  
- Enhanced the baseline in several ways:
  - Added hyperparameter sweeps (learning rates, target network update intervals)  
  - Tested stability variants (e.g. double DQN, dueling architecture)  
  - Evaluated performance across multiple random seeds to measure variance  
  - Analyzed learning curves, reward distributions, convergence speed  
- Submitted a detailed report documenting:
  - Methodology and algorithm choices  
  - Comparisons of performance across different variants  
  - Discussion of failures, variance, and improvement pathways  

---
