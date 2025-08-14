import numpy as np
import pickle
from typing import Tuple, Dict, Any, Optional
import random


class QLearningAgent:
    """
    Q-Learning Agent for Dice Games
    
    Uses tabular Q-learning to learn optimal strategies for dice-based games.
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dictionary mapping (state, action) -> Q-value
        self.q_table = {}
        
        # Training statistics
        self.training_rewards = []
        self.episode_rewards = []
        self.win_rates = []
    
    def get_q_value(self, state: Tuple, action: int) -> float:
        """Get Q-value for state-action pair"""
        return self.q_table.get((state, action), 0.0)
    
    def set_q_value(self, state: Tuple, action: int, value: float):
        """Set Q-value for state-action pair"""
        self.q_table[(state, action)] = value
    
    def choose_action(self, state: Tuple, valid_actions: Optional[list] = None, 
                     epsilon: Optional[float] = None) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current game state
            valid_actions: List of valid actions (if None, assumes all actions valid)
            epsilon: Exploration rate (if None, uses instance epsilon)
            
        Returns:
            action: Chosen action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if valid_actions is None:
            # For Pig game, actions are always [0, 1] (hold, roll)
            valid_actions = [0, 1]
        
        # Epsilon-greedy policy
        if random.random() < epsilon:
            # Exploration: choose random action
            return random.choice(valid_actions)
        else:
            # Exploitation: choose best action
            return self._get_best_action(state, valid_actions)
    
    def _get_best_action(self, state: Tuple, valid_actions: list) -> int:
        """Get the best action according to Q-values"""
        best_action = valid_actions[0]
        best_q_value = self.get_q_value(state, best_action)
        
        for action in valid_actions[1:]:
            q_value = self.get_q_value(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action
    
    def update(self, state: Tuple, action: int, reward: float, 
               next_state: Tuple, next_valid_actions: Optional[list] = None):
        """
        Update Q-value using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_valid_actions: Valid actions in next state
        """
        if next_valid_actions is None:
            next_valid_actions = [0, 1]  # Default for Pig game
        
        # Current Q-value
        current_q = self.get_q_value(state, action)
        
        # Max Q-value for next state
        max_next_q = max(self.get_q_value(next_state, a) for a in next_valid_actions)
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.set_q_value(state, action, new_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self, filename: str):
        """Save Q-table to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self, filename: str):
        """Load Q-table from file"""
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
    
    def get_policy(self, state: Tuple, valid_actions: Optional[list] = None) -> Dict[int, float]:
        """
        Get policy (action probabilities) for a state
        
        Args:
            state: Current state
            valid_actions: List of valid actions
            
        Returns:
            policy: Dictionary mapping actions to probabilities
        """
        if valid_actions is None:
            valid_actions = [0, 1]
        
        q_values = [self.get_q_value(state, action) for action in valid_actions]
        
        # Convert to probabilities using softmax
        exp_q = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        probabilities = exp_q / np.sum(exp_q)
        
        return dict(zip(valid_actions, probabilities))
    
    def get_state_value(self, state: Tuple, valid_actions: Optional[list] = None) -> float:
        """
        Get state value (max Q-value for the state)
        
        Args:
            state: Current state
            valid_actions: List of valid actions
            
        Returns:
            value: State value
        """
        if valid_actions is None:
            valid_actions = [0, 1]
        
        return max(self.get_q_value(state, action) for action in valid_actions)
    
    def get_q_table_size(self) -> int:
        """Get the number of state-action pairs in Q-table"""
        return len(self.q_table)
    
    def reset_training_stats(self):
        """Reset training statistics"""
        self.training_rewards = []
        self.episode_rewards = []
        self.win_rates = []
    
    def add_training_reward(self, reward: float):
        """Add reward to training statistics"""
        self.training_rewards.append(reward)
    
    def add_episode_reward(self, total_reward: float):
        """Add episode total reward to statistics"""
        self.episode_rewards.append(total_reward)
    
    def add_win_rate(self, win_rate: float):
        """Add win rate to statistics"""
        self.win_rates.append(win_rate)
    
    def get_training_stats(self) -> Dict[str, list]:
        """Get training statistics"""
        return {
            'training_rewards': self.training_rewards,
            'episode_rewards': self.episode_rewards,
            'win_rates': self.win_rates
        }


class QLearningAgentWithExperienceReplay(QLearningAgent):
    """
    Q-Learning Agent with Experience Replay
    
    Stores experiences and samples from them for more stable learning.
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 replay_buffer_size: int = 10000, batch_size: int = 32):
        super().__init__(learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min)
        
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
    
    def store_experience(self, state: Tuple, action: int, reward: float, 
                        next_state: Tuple, done: bool, next_valid_actions: Optional[list] = None):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            next_valid_actions: Valid actions in next state
        """
        experience = (state, action, reward, next_state, done, next_valid_actions)
        self.replay_buffer.append(experience)
        
        # Remove oldest experience if buffer is full
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)
    
    def sample_experiences(self) -> list:
        """Sample a batch of experiences from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return self.replay_buffer
        
        return random.sample(self.replay_buffer, self.batch_size)
    
    def replay(self):
        """Update Q-values using a batch of experiences"""
        experiences = self.sample_experiences()
        
        for state, action, reward, next_state, done, next_valid_actions in experiences:
            if next_valid_actions is None:
                next_valid_actions = [0, 1]
            
            # Current Q-value
            current_q = self.get_q_value(state, action)
            
            if done:
                # Terminal state
                new_q = current_q + self.learning_rate * (reward - current_q)
            else:
                # Non-terminal state
                max_next_q = max(self.get_q_value(next_state, a) for a in next_valid_actions)
                new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            
            # Update Q-table
            self.set_q_value(state, action, new_q) 