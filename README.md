# Grid World Q-Learning Agent

## Overview
This project implements a simple **Q-Learning reinforcement learning agent** that navigates a 10x10 grid world. The agent learns over time to maximize rewards and avoid penalties by interacting with its environment through trial and error.

The system visualizes learning progress, behavior evolution, and path changes from early training to later stages.

---

## How It Works

The environment is a **10x10 grid** containing:
- 🟢 Reward cells (+1)
- 🔴 Penalty cells (-1)
- 🔵 A fixed starting position (center of grid at (5,5))

The agent:
- Starts with no knowledge (empty Q-table)
- Learns using **Q-learning**
- Balances exploration vs exploitation using an epsilon-greedy strategy
- Updates its policy over **500 episodes**

---

## Key Concepts

- **Q-Learning**: Updates a Q-table to estimate the value of actions in each state.
- **Exploration vs Exploitation**: Controlled by epsilon decay.
- **Reward System**: Encourages visiting reward cells and avoiding penalty cells.
- **Episodes**: Each episode is one full run through the environment.
- **Learning Rate (α)**: Controls how fast new experiences overwrite old knowledge.
- **Discount Factor (γ)**: Determines how much future rewards matter.

---

## Parameters

| Parameter | Meaning |
|----------|--------|
| GRID_SIZE | Size of environment (10x10) |
| EPISODES | Number of training runs (500) |
| MAX_STEPS | Max steps per episode (200) |
| LEARNING_RATE | How fast the agent learns |
| DISCOUNT | Importance of future rewards |
| EPSILON_START | Initial randomness (exploration) |
| EPSILON_END | Minimum randomness allowed |
| EPSILON_DECAY | Rate of exploration reduction |

---

## Outputs

After training, the program generates:

### 📊 Learning Visualizations
- Reward progression over time
- Heatmap of visited cells
- Environment layout (rewards, penalties, start)
- Behavioral snapshots every 50 episodes

### 🧠 Agent Behavior Comparison
- Path taken in **first episode**
- Path taken in **final episode**

### 📁 Saved File
- `emergent_behavior.png` → full visualization dashboard

---

## File Structure
