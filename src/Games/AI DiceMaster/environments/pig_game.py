import numpy as np
import random
from typing import Tuple, Dict, Any, Optional


class PigGame:
    """
    Pig Game Environment for Reinforcement Learning
    
    Rules:
    - Players take turns rolling a die
    - On each turn, a player can roll or hold
    - Rolling adds points to turn total, but rolling 1 loses all turn points
    - First player to reach 100 points wins
    """
    
    def __init__(self, target_score: int = 100):
        self.target_score = target_score
        self.reset()
        
    def reset(self) -> Tuple[Tuple[int, int, int], Dict[str, Any]]:
        """
        Reset the game to initial state
        
        Returns:
            state: (player_score, opponent_score, turn_total)
            info: Additional game information
        """
        self.player_score = 0
        self.opponent_score = 0
        self.current_score = 0
        self.turn_total = 0
        self.done = False
        self.winner = None
        
        # Randomly decide who goes first
        self.player_turn = random.choice([True, False])
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]:
        """
        Take an action in the environment
        
        Args:
            action: 0 for hold, 1 for roll
            
        Returns:
            state: Current game state
            reward: Reward for the action
            done: Whether the game is finished
            info: Additional information
        """
        if self.done:
            raise ValueError("Game is already finished. Call reset() to start a new game.")
        
        reward = 0
        
        if action == 1:  # Roll
            die_roll = random.randint(1, 6)
            
            if die_roll == 1:
                # Lose turn total
                lost_points = self.turn_total
                self.turn_total = 0
                reward = -lost_points  # Penalty for losing points
                self._switch_turns()
            else:
                # Add to turn total
                self.turn_total += die_roll
                reward = die_roll  # Small positive reward for each point
                
        elif action == 0:  # Hold
            # Add turn total to current player's score
            if self.player_turn:
                self.player_score += self.turn_total
                reward = self.turn_total  # Reward for points gained
            else:
                self.opponent_score += self.turn_total
                reward = -self.turn_total  # Penalty for opponent gaining points
            
            self.turn_total = 0
            self._switch_turns()
        
        # Check if game is over
        if self.player_score >= self.target_score:
            self.done = True
            self.winner = "player"
            reward += 100  # Large reward for winning
        elif self.opponent_score >= self.target_score:
            self.done = True
            self.winner = "opponent"
            reward -= 100  # Large penalty for losing
        
        state = self._get_state()
        info = self._get_info()
        
        return state, reward, self.done, info
    
    def _get_state(self) -> Tuple[int, int, int]:
        """Get current state as (player_score, opponent_score, turn_total)"""
        return (self.player_score, self.opponent_score, self.turn_total)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional game information"""
        return {
            "player_turn": self.player_turn,
            "winner": self.winner,
            "target_score": self.target_score
        }
    
    def _switch_turns(self):
        """Switch turns between player and opponent"""
        self.player_turn = not self.player_turn
    
    def render(self):
        """Render the current game state"""
        print(f"Player Score: {self.player_score}")
        print(f"Opponent Score: {self.opponent_score}")
        print(f"Turn Total: {self.turn_total}")
        print(f"Player's Turn: {self.player_turn}")
        if self.done:
            print(f"Game Over! Winner: {self.winner}")
        print("-" * 30)
    
    def get_valid_actions(self) -> list:
        """Get list of valid actions (always [0, 1] for hold/roll)"""
        return [0, 1]
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space"""
        return 2
    
    def get_state_space_size(self) -> Tuple[int, int, int]:
        """Get the size of the state space (max scores + max turn total)"""
        return (self.target_score + 1, self.target_score + 1, 100)  # Reasonable upper bound for turn total


class RandomOpponent:
    """Simple random opponent for training"""
    
    def __init__(self, hold_threshold: int = 20):
        self.hold_threshold = hold_threshold
    
    def choose_action(self, state: Tuple[int, int, int]) -> int:
        """
        Choose action based on current state
        
        Args:
            state: (player_score, opponent_score, turn_total)
            
        Returns:
            action: 0 for hold, 1 for roll
        """
        _, _, turn_total = state
        
        # Hold if turn total reaches threshold, otherwise roll
        if turn_total >= self.hold_threshold:
            return 0  # Hold
        else:
            return 1  # Roll


class StrategicOpponent:
    """Strategic opponent that uses basic strategy"""
    
    def __init__(self):
        pass
    
    def choose_action(self, state: Tuple[int, int, int]) -> int:
        """
        Choose action based on current state using basic strategy
        
        Args:
            state: (player_score, opponent_score, turn_total)
            
        Returns:
            action: 0 for hold, 1 for roll
        """
        player_score, opponent_score, turn_total = state
        
        # Basic strategy: hold if turn total is high enough
        # or if opponent is close to winning
        if turn_total >= 25:
            return 0  # Hold
        
        # If opponent is close to winning, be more aggressive
        if opponent_score >= 80 and turn_total >= 15:
            return 0  # Hold
        
        # If we're far behind, be more aggressive
        if player_score < opponent_score - 30 and turn_total >= 10:
            return 1  # Roll
        
        return 1  # Default to rolling 