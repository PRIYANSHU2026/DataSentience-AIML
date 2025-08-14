# 🎲 AI DiceMaster: Reinforcement Learning for Dice Games

> **Author:** [@SK8-infi](https://github.com/SK8-infi)  
> **License:** MIT License  
> **Version:** 1.0.0

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

**AI DiceMaster** is a comprehensive reinforcement learning framework for training AI agents to play dice-based games. Built with modern Python and PyTorch, it provides a complete ecosystem for experimenting with Q-learning and Deep Q-Networks (DQN) on stochastic game environments.

The project demonstrates how AI agents can learn optimal strategies through trial and error, balancing exploration vs exploitation, and achieving competitive performance against various opponents.

### 🧠 Key Concepts

- **Reinforcement Learning**: Agents learn optimal strategies through trial and error
- **Q-Learning**: Tabular approach for discrete state-action spaces
- **Deep Q-Networks (DQN)**: Neural network approximation for complex environments
- **Experience Replay**: Stabilizes learning by storing and sampling past experiences
- **Epsilon-Greedy Policy**: Balances exploration and exploitation during training

## ✨ Features

### 🎮 Game Environments
- **Pig Game**: Classic dice game with hold/roll mechanics
- **Ludo Game**: Simplified 4-player board game
- **Modular Design**: Easy to add new games

### 🤖 AI Agents
- **Q-Learning Agent**: Tabular Q-learning with experience replay
- **DQN Agent**: Deep Q-Network with PyTorch
- **Configurable Parameters**: Learning rates, exploration, network architecture

### 📊 Training & Evaluation
- **Comprehensive Metrics**: Win rates, average scores, action distributions
- **Visualization Tools**: Training curves, agent comparisons, learning progress
- **Multiple Opponents**: Random, strategic, and custom opponents

### 🧪 Testing & Quality
- **31 Test Cases**: Comprehensive test suite
- **Modular Architecture**: Clean separation of concerns
- **Documentation**: Inline docs and examples

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/SK8-infi/ai-dicemaster.git
cd ai-dicemaster

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run tests to verify everything works
python -m pytest tests/ -v

# Run the demo
python main.py
```

## 🎯 Quick Start

### Basic Training Example

```python
from training.train_q_learning import train_q_learning_agent
from evaluation.evaluate_agent import evaluate_agent_detailed

# Train a Q-learning agent
agent = train_q_learning_agent(
    episodes=1000,
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=0.1,
    opponent_type="random"
)

# Evaluate the trained agent
stats = evaluate_agent_detailed(agent, episodes=100)
print(f"Win Rate: {stats['win_rate']:.2%}")
```

### DQN Training Example

```python
from training.train_dqn import train_dqn_agent

# Train a DQN agent
agent = train_dqn_agent(
    episodes=1000,
    learning_rate=0.001,
    discount_factor=0.9,
    epsilon=1.0,
    opponent_type="random"
)
```

## 📖 Usage Examples

### 1. Playing Against a Trained Agent

```python
from environments.pig_game import PigGame
from agents.q_learning_agent import QLearningAgent

# Load trained agent
agent = QLearningAgent()
agent.load_q_table('trained_agent.pkl')

# Play a game
game = PigGame(target_score=100)
state, info = game.reset()

while not game.done:
    if game.player_turn:
        # Agent's turn
        action = agent.choose_action(state, epsilon=0)  # No exploration
        state, reward, done, info = game.step(action)
        print(f"Agent chose: {'HOLD' if action == 0 else 'ROLL'}")
    else:
        # Your turn (simulated)
        action = int(input("Your turn (0=HOLD, 1=ROLL): "))
        state, reward, done, info = game.step(action)
    
    game.render()
```

### 2. Agent Comparison

```python
from evaluation.evaluate_agent import compare_agents
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent

# Create agents
agents = {
    "Q-Learning": QLearningAgent(),
    "DQN": DQNAgent(state_size=3, action_size=2)
}

# Train and compare
comparison = compare_agents(agents, episodes=1000)
```

### 3. Custom Game Environment

```python
from environments.pig_game import PigGame

class CustomPigGame(PigGame):
    def __init__(self, target_score=100, bonus_threshold=25):
        super().__init__(target_score)
        self.bonus_threshold = bonus_threshold
    
    def step(self, action):
        state, reward, done, info = super().step(action)
        
        # Add bonus reward for high turn totals
        if self.turn_total >= self.bonus_threshold:
            reward += 5
        
        return state, reward, done, info
```

### 4. Training with Custom Parameters

```python
from training.train_q_learning import train_q_learning_agent

# Advanced training configuration
agent = train_q_learning_agent(
    episodes=5000,
    learning_rate=0.15,
    discount_factor=0.95,
    epsilon=0.2,
    epsilon_decay=0.999,
    epsilon_min=0.005,
    opponent_type="strategic",
    target_score=50,
    eval_freq=500
)
```

## 📚 API Reference

### Game Environments

#### PigGame
```python
class PigGame:
    def __init__(self, target_score: int = 100)
    def reset() -> Tuple[Tuple[int, int, int], Dict[str, Any]]
    def step(action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]
    def render()
    def get_valid_actions() -> List[int]
```

**State Space:** `(player_score, opponent_score, turn_total)`  
**Action Space:** `[0, 1]` (hold, roll)

#### LudoGame
```python
class LudoGame:
    def __init__(self, board_size: int = 52, home_size: int = 6)
    def reset() -> Tuple[Tuple[int, ...], Dict[str, Any]]
    def step(action: int) -> Tuple[Tuple[int, ...], float, bool, Dict[str, Any]]
    def get_valid_actions() -> List[int]
```

### AI Agents

#### QLearningAgent
```python
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    def choose_action(state, valid_actions=None, epsilon=None) -> int
    def update(state, action, reward, next_state, next_valid_actions=None)
    def save_q_table(filename: str)
    def load_q_table(filename: str)
    def get_training_stats() -> Dict[str, List[float]]
```

#### DQNAgent
```python
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, ...)
    def choose_action(state, valid_actions=None, epsilon=None) -> int
    def store_experience(state, action, reward, next_state, done)
    def train()
    def save_model(filename: str)
    def load_model(filename: str)
```

### Training Functions

```python
def train_q_learning_agent(episodes, learning_rate, discount_factor, epsilon, ...)
def train_dqn_agent(episodes, learning_rate, discount_factor, epsilon, ...)
def evaluate_agent(agent, episodes, opponent_type, target_score)
def evaluate_agent_detailed(agent, episodes, opponent_type, target_score)
```

### Visualization Functions

```python
def plot_training_stats(stats: Dict[str, List[float]], title: str)
def plot_agent_comparison(comparison: Dict[str, Dict], title: str)
def plot_learning_curves(agents: Dict[str, Any], episodes, opponent_type)
```

## 🏗️ Architecture

### Project Structure
```
ai-dicemaster/
├── environments/          # Game environments
│   ├── pig_game.py      # Pig game implementation
│   └── ludo_game.py     # Ludo game implementation
├── agents/              # RL agents
│   ├── q_learning_agent.py  # Q-learning implementation
│   └── dqn_agent.py        # DQN implementation
├── training/            # Training scripts
│   ├── train_q_learning.py  # Q-learning training
│   └── train_dqn.py        # DQN training
├── evaluation/          # Evaluation tools
│   └── evaluate_agent.py   # Agent evaluation
├── utils/              # Utilities
│   └── plotting.py        # Visualization tools
├── tests/              # Test suite
│   └── test_environment.py # Comprehensive tests
├── main.py             # Demo script
├── requirements.txt    # Dependencies
└── README.md          # This file
```

### Design Patterns

1. **Environment Interface**: Standard RL environment with `reset()`, `step()`, `render()`
2. **Agent Interface**: Consistent API across different RL algorithms
3. **Modular Training**: Separate training scripts for different algorithms
4. **Comprehensive Evaluation**: Multiple metrics and visualization options

### Key Components

#### Game Environments
- **State Representation**: Efficient state encoding for RL
- **Action Space**: Discrete actions with validation
- **Reward Shaping**: Carefully designed rewards for learning
- **Opponent Models**: Various opponent strategies for training

#### AI Agents
- **Q-Learning**: Tabular approach with experience replay
- **DQN**: Neural network with target networks and experience replay
- **Exploration Strategies**: Epsilon-greedy with decay
- **Training Statistics**: Comprehensive logging and metrics

#### Evaluation Framework
- **Multiple Metrics**: Win rates, scores, action distributions
- **Opponent Variety**: Random, strategic, and custom opponents
- **Visualization**: Rich plotting and comparison tools

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_environment.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Coverage
- **Game Environments**: State transitions, actions, rewards
- **AI Agents**: Learning, action selection, model persistence
- **Training**: Training loops, evaluation, statistics
- **Integration**: End-to-end training and evaluation

## 📈 Performance

### Typical Results

| Agent | Win Rate vs Random | Win Rate vs Strategic | Training Time |
|-------|-------------------|---------------------|---------------|
| Q-Learning | 85-90% | 70-75% | ~5 minutes |
| DQN | 80-85% | 65-70% | ~15 minutes |

### Training Parameters

**Q-Learning:**
- Learning Rate: 0.1
- Discount Factor: 0.9
- Epsilon: 0.1 (decay to 0.01)
- Episodes: 10,000

**DQN:**
- Learning Rate: 0.001
- Discount Factor: 0.9
- Epsilon: 1.0 (decay to 0.01)
- Batch Size: 32
- Replay Buffer: 10,000
- Episodes: 10,000

## 🔧 Configuration

### Environment Parameters
```python
# Pig Game
target_score = 100          # Score to win
bonus_threshold = 25        # Bonus for high turn totals

# Ludo Game
board_size = 52            # Board length
home_size = 6              # Home area size
num_players = 4            # Number of players
tokens_per_player = 4      # Tokens per player
```

### Agent Parameters
```python
# Q-Learning
learning_rate = 0.1        # Q-learning rate
discount_factor = 0.9      # Future reward discount
epsilon = 0.1              # Exploration rate
epsilon_decay = 0.995      # Epsilon decay rate
epsilon_min = 0.01         # Minimum epsilon

# DQN
learning_rate = 0.001      # Neural network learning rate
batch_size = 32            # Training batch size
replay_buffer_size = 10000 # Experience replay buffer
target_update_freq = 100   # Target network update frequency
hidden_sizes = [128, 64]   # Neural network architecture
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/ai-dicemaster.git
cd ai-dicemaster

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8
```

### Code Style
```bash
# Format code
black .

# Lint code
flake8 .

# Run tests
python -m pytest tests/ -v
```

## 📞 Contact

- **Author**: [@SK8-infi](https://github.com/SK8-infi)
- **Project**: [AI DiceMaster](https://github.com/SK8-infi/ai-dicemaster)
---

**Made with ❤️ by [@SK8-infi](https://github.com/SK8-infi)**

*"The best way to predict the future is to invent it." - Alan Kay* 