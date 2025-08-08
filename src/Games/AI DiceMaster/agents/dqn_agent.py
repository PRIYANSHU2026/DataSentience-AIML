import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Tuple, Dict, Any, Optional, List
import pickle


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Dice Games
    
    Neural network that approximates Q-values for state-action pairs.
    """
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = None):
        super(DQNNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)


class DQNAgent:
    """
    Deep Q-Network Agent for Dice Games
    
    Uses a neural network to approximate Q-values and experience replay
    for stable learning.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001, discount_factor: float = 0.9,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 replay_buffer_size: int = 10000, batch_size: int = 32,
                 target_update_freq: int = 100, hidden_sizes: List[int] = None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Training statistics
        self.training_rewards = []
        self.episode_rewards = []
        self.win_rates = []
        self.losses = []
        
        # Update counter
        self.update_counter = 0
        
        # Initialize target network
        self._update_target_network()
    
    def _state_to_tensor(self, state: Tuple) -> torch.Tensor:
        """Convert state tuple to tensor"""
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def _get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all actions in a state"""
        return self.q_network(state)
    
    def choose_action(self, state: Tuple, valid_actions: Optional[List[int]] = None,
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
            valid_actions = list(range(self.action_size))
        
        # Epsilon-greedy policy
        if random.random() < epsilon:
            # Exploration: choose random action
            return random.choice(valid_actions)
        else:
            # Exploitation: choose best action
            return self._get_best_action(state, valid_actions)
    
    def _get_best_action(self, state: Tuple, valid_actions: List[int]) -> int:
        """Get the best action according to Q-values"""
        state_tensor = self._state_to_tensor(state)
        q_values = self._get_q_values(state_tensor)
        
        # Mask invalid actions with large negative values
        mask = torch.ones(self.action_size, dtype=torch.bool)
        mask[valid_actions] = False
        q_values[0, mask] = float('-inf')
        
        return q_values.argmax().item()
    
    def store_experience(self, state: Tuple, action: int, reward: float,
                        next_state: Tuple, done: bool, next_valid_actions: Optional[List[int]] = None):
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
    
    def _sample_experiences(self) -> List[Tuple]:
        """Sample a batch of experiences from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return list(self.replay_buffer)
        
        return random.sample(self.replay_buffer, self.batch_size)
    
    def _update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self):
        """Train the network using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample experiences
        experiences = self._sample_experiences()
        
        # Prepare batch
        states = torch.FloatTensor([exp[0] for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp[4] for exp in experiences]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (from target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()
        
        # Store loss
        self.losses.append(loss.item())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filename: str):
        """Save model to file"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load_model(self, filename: str):
        """Load model from file"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
    
    def get_policy(self, state: Tuple, valid_actions: Optional[List[int]] = None) -> Dict[int, float]:
        """
        Get policy (action probabilities) for a state
        
        Args:
            state: Current state
            valid_actions: List of valid actions
            
        Returns:
            policy: Dictionary mapping actions to probabilities
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
        
        state_tensor = self._state_to_tensor(state)
        q_values = self._get_q_values(state_tensor).squeeze()
        
        # Mask invalid actions
        mask = torch.ones(self.action_size, dtype=torch.bool)
        mask[valid_actions] = False
        q_values[mask] = float('-inf')
        
        # Convert to probabilities using softmax
        probabilities = F.softmax(q_values, dim=0)
        
        return {action: probabilities[action].item() for action in valid_actions}
    
    def get_state_value(self, state: Tuple, valid_actions: Optional[List[int]] = None) -> float:
        """
        Get state value (max Q-value for the state)
        
        Args:
            state: Current state
            valid_actions: List of valid actions
            
        Returns:
            value: State value
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
        
        state_tensor = self._state_to_tensor(state)
        q_values = self._get_q_values(state_tensor).squeeze()
        
        # Mask invalid actions
        mask = torch.ones(self.action_size, dtype=torch.bool)
        mask[valid_actions] = False
        q_values[mask] = float('-inf')
        
        return q_values.max().item()
    
    def reset_training_stats(self):
        """Reset training statistics"""
        self.training_rewards = []
        self.episode_rewards = []
        self.win_rates = []
        self.losses = []
    
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
            'win_rates': self.win_rates,
            'losses': self.losses
        }
    
    def get_replay_buffer_size(self) -> int:
        """Get the number of experiences in replay buffer"""
        return len(self.replay_buffer) 