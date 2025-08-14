import numpy as np
import random
from typing import Tuple, Dict, Any, List


class LudoGame:
    """
    Simplified Ludo Game Environment for Reinforcement Learning
    
    Rules:
    - 4 players, each with 4 tokens
    - Roll 1-6 sided die
    - Move tokens around board
    - Get all tokens home to win
    - Simplified version for RL training
    """
    
    def __init__(self, board_size: int = 52, home_size: int = 6):
        self.board_size = board_size
        self.home_size = home_size
        self.num_players = 4
        self.tokens_per_player = 4
        self.reset()
    
    def reset(self) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        """
        Reset the game to initial state
        
        Returns:
            state: (player_positions, opponent_positions, die_roll, available_actions)
            info: Additional game information
        """
        # Initialize all tokens at start position (-1 means not started)
        self.positions = [[-1] * self.tokens_per_player for _ in range(self.num_players)]
        self.home_tokens = [0] * self.num_players
        self.current_player = 0
        self.done = False
        self.winner = None
        self.last_die_roll = 0
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, Dict[str, Any]]:
        """
        Take an action in the environment
        
        Args:
            action: Token index to move (0-3) or -1 for no move
            
        Returns:
            state: Current game state
            reward: Reward for the action
            done: Whether the game is finished
            info: Additional information
        """
        if self.done:
            raise ValueError("Game is already finished. Call reset() to start a new game.")
        
        reward = 0
        
        # Roll die
        die_roll = random.randint(1, 6)
        self.last_die_roll = die_roll
        
        # Handle action
        if action >= 0 and action < self.tokens_per_player:
            # Move token
            old_pos = self.positions[self.current_player][action]
            new_pos = self._move_token(old_pos, die_roll)
            
            if new_pos != old_pos:
                self.positions[self.current_player][action] = new_pos
                
                # Check if token reached home
                if new_pos >= self.board_size:
                    self.home_tokens[self.current_player] += 1
                    reward += 10  # Reward for getting token home
                
                # Check if token captured opponent
                captured = self._check_capture(new_pos)
                if captured:
                    reward += 5  # Reward for capturing
        
        # Check if current player won
        if self.home_tokens[self.current_player] == self.tokens_per_player:
            self.done = True
            self.winner = self.current_player
            reward += 100  # Large reward for winning
        
        # Switch to next player
        self.current_player = (self.current_player + 1) % self.num_players
        
        state = self._get_state()
        info = self._get_info()
        
        return state, reward, self.done, info
    
    def _move_token(self, current_pos: int, die_roll: int) -> int:
        """Move token based on die roll"""
        if current_pos == -1:  # Token not started
            if die_roll == 6:
                return 0  # Start token
            else:
                return -1  # Can't start without 6
        else:
            new_pos = current_pos + die_roll
            if new_pos >= self.board_size:
                return self.board_size  # Token reached home
            else:
                return new_pos
    
    def _check_capture(self, position: int) -> bool:
        """Check if token captured an opponent token"""
        for player in range(self.num_players):
            if player != self.current_player:
                for token_idx in range(self.tokens_per_player):
                    if self.positions[player][token_idx] == position:
                        # Capture opponent token
                        self.positions[player][token_idx] = -1
                        return True
        return False
    
    def _get_state(self) -> Tuple[int, ...]:
        """Get current state as flattened array"""
        state = []
        
        # Add all player positions
        for player in range(self.num_players):
            state.extend(self.positions[player])
        
        # Add home tokens
        state.extend(self.home_tokens)
        
        # Add current player and die roll
        state.append(self.current_player)
        state.append(self.last_die_roll)
        
        return tuple(state)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional game information"""
        return {
            "current_player": self.current_player,
            "winner": self.winner,
            "home_tokens": self.home_tokens.copy(),
            "positions": [pos.copy() for pos in self.positions]
        }
    
    def render(self):
        """Render the current game state"""
        print(f"Current Player: {self.current_player}")
        print(f"Last Die Roll: {self.last_die_roll}")
        print("Player Positions:")
        for i, positions in enumerate(self.positions):
            print(f"  Player {i}: {positions} (Home: {self.home_tokens[i]})")
        if self.done:
            print(f"Game Over! Winner: Player {self.winner}")
        print("-" * 30)
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions for current player"""
        valid_actions = []
        
        for token_idx in range(self.tokens_per_player):
            pos = self.positions[self.current_player][token_idx]
            
            # Can always try to move a token if it's on the board
            if pos >= 0:
                valid_actions.append(token_idx)
            # Can start a new token if we have a 6
            elif self.last_die_roll == 6:
                valid_actions.append(token_idx)
        
        # Always allow no move
        valid_actions.append(-1)
        
        return valid_actions
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space"""
        return self.tokens_per_player + 1  # 4 tokens + no move
    
    def get_state_space_size(self) -> Tuple[int, ...]:
        """Get the size of the state space"""
        # Positions: 4 players * 4 tokens * (board_size + 1 for -1)
        # Home tokens: 4 players
        # Current player: 4
        # Die roll: 6
        return tuple([self.board_size + 1] * 16 + [self.tokens_per_player] * 4 + [4, 6])


class RandomLudoOpponent:
    """Simple random opponent for Ludo training"""
    
    def __init__(self):
        pass
    
    def choose_action(self, state: Tuple[int, ...], valid_actions: List[int]) -> int:
        """
        Choose action based on current state
        
        Args:
            state: Current game state
            valid_actions: List of valid actions
            
        Returns:
            action: Token index to move or -1 for no move
        """
        return random.choice(valid_actions)


class StrategicLudoOpponent:
    """Strategic opponent for Ludo that prioritizes getting tokens home"""
    
    def __init__(self):
        pass
    
    def choose_action(self, state: Tuple[int, ...], valid_actions: List[int]) -> int:
        """
        Choose action based on current state using basic strategy
        
        Args:
            state: Current game state
            valid_actions: List of valid actions
            
        Returns:
            action: Token index to move or -1 for no move
        """
        # Extract positions from state
        positions = []
        for i in range(16):  # 4 players * 4 tokens
            positions.append(state[i])
        
        # Group positions by player
        player_positions = []
        for player in range(4):
            player_positions.append(positions[player*4:(player+1)*4])
        
        # Get current player (assuming it's the last but one element)
        current_player = state[-2]
        
        # Strategy: prioritize tokens closer to home
        best_action = -1
        best_score = -1
        
        for action in valid_actions:
            if action == -1:
                continue
            
            token_pos = player_positions[current_player][action]
            if token_pos >= 0:
                # Score based on how close to home
                distance_to_home = 52 - token_pos
                score = 1.0 / (distance_to_home + 1)  # Closer = higher score
                
                if score > best_score:
                    best_score = score
                    best_action = action
        
        return best_action if best_action != -1 else random.choice(valid_actions) 