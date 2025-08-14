import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import patch
import tempfile
import pickle

from environments.pig_game import PigGame, RandomOpponent, StrategicOpponent
from environments.ludo_game import LudoGame, RandomLudoOpponent, StrategicLudoOpponent
from agents.q_learning_agent import QLearningAgent, QLearningAgentWithExperienceReplay
from agents.dqn_agent import DQNAgent, DQNNetwork
from evaluation.evaluate_agent import evaluate_agent, evaluate_agent_detailed


class TestPigGame:
    """Test cases for the Pig game environment"""
    
    def test_game_initialization(self):
        """Test that the game initializes correctly"""
        game = PigGame(target_score=50)
        assert game.target_score == 50
        assert game.player_score == 0
        assert game.opponent_score == 0
        assert game.turn_total == 0
        assert not game.done
        assert game.winner is None
    
    def test_game_reset(self):
        """Test that the game resets correctly"""
        game = PigGame()
        game.player_score = 50
        game.opponent_score = 30
        game.turn_total = 15
        game.done = True
        game.winner = "player"
        
        state, info = game.reset()
        
        assert game.player_score == 0
        assert game.opponent_score == 0
        assert game.turn_total == 0
        assert not game.done
        assert game.winner is None
        assert len(state) == 3
        assert isinstance(info, dict)
    
    def test_hold_action(self):
        """Test the hold action"""
        game = PigGame()
        state, info = game.reset()
        game.turn_total = 10
        game.player_turn = True
        
        next_state, reward, done, info = game.step(0)  # Hold action
        
        assert game.player_score == 10
        assert game.turn_total == 0
        assert not game.player_turn
        assert reward == 10
        assert not done
    
    def test_roll_action_success(self):
        """Test successful roll action"""
        game = PigGame()
        state, info = game.reset()
        game.player_turn = True
        
        with patch('random.randint', return_value=4):
            next_state, reward, done, info = game.step(1)  # Roll action
        
        assert game.turn_total == 4
        assert game.player_turn  # Still player's turn
        assert reward == 4
        assert not done
    
    def test_roll_action_bust(self):
        """Test roll action that results in bust (rolling 1)"""
        game = PigGame()
        state, info = game.reset()
        game.turn_total = 15
        game.player_turn = True
        
        with patch('random.randint', return_value=1):
            next_state, reward, done, info = game.step(1)  # Roll action
        
        assert game.turn_total == 0
        assert not game.player_turn  # Turn switched
        assert reward == -15  # Penalty for losing points
        assert not done
    
    def test_game_winning(self):
        """Test that the game ends when a player reaches target score"""
        game = PigGame(target_score=20)
        state, info = game.reset()
        game.player_score = 15
        game.turn_total = 10
        game.player_turn = True
        
        next_state, reward, done, info = game.step(0)  # Hold action
        
        assert game.player_score == 25
        assert done
        assert game.winner == "player"
        assert reward == 110  # 10 points + 100 for winning
    
    def test_get_valid_actions(self):
        """Test that valid actions are returned correctly"""
        game = PigGame()
        valid_actions = game.get_valid_actions()
        assert valid_actions == [0, 1]
    
    def test_get_state_space_size(self):
        """Test state space size calculation"""
        game = PigGame(target_score=100)
        state_space = game.get_state_space_size()
        assert state_space == (101, 101, 100)


class TestLudoGame:
    """Test cases for the Ludo game environment"""
    
    def test_game_initialization(self):
        """Test that the Ludo game initializes correctly"""
        game = LudoGame()
        assert game.board_size == 52
        assert game.num_players == 4
        assert game.tokens_per_player == 4
        assert len(game.positions) == 4
        assert len(game.home_tokens) == 4
    
    def test_game_reset(self):
        """Test that the Ludo game resets correctly"""
        game = LudoGame()
        game.positions[0] = [10, 20, 30, 40]
        game.home_tokens[0] = 2
        game.current_player = 2
        game.done = True
        game.winner = 1
        
        state, info = game.reset()
        
        assert all(pos == -1 for pos in game.positions[0])
        assert game.home_tokens[0] == 0
        assert game.current_player == 0
        assert not game.done
        assert game.winner is None
    
    def test_move_token_start(self):
        """Test starting a token with a 6"""
        game = LudoGame()
        game.current_player = 0
        
        with patch('random.randint', return_value=6):
            next_state, reward, done, info = game.step(0)  # Move token 0
        
        assert game.positions[0][0] == 0  # Token started
        assert reward == 0  # No reward for starting
    
    def test_move_token_normal(self):
        """Test normal token movement"""
        game = LudoGame()
        game.current_player = 0
        game.positions[0][0] = 10  # Token already on board
        
        with patch('random.randint', return_value=3):
            next_state, reward, done, info = game.step(0)  # Move token 0
        
        assert game.positions[0][0] == 13  # Moved 3 spaces
        assert reward == 0  # No reward for normal movement
    
    def test_token_reaches_home(self):
        """Test token reaching home"""
        game = LudoGame()
        game.current_player = 0
        game.positions[0][0] = 50  # Close to home
        
        with patch('random.randint', return_value=5):
            next_state, reward, done, info = game.step(0)  # Move token 0
        
        assert game.positions[0][0] == 52  # Reached home
        assert game.home_tokens[0] == 1
        assert reward == 10  # Reward for reaching home
    
    def test_player_wins(self):
        """Test that a player wins when all tokens reach home"""
        game = LudoGame()
        game.current_player = 0
        game.home_tokens[0] = 3  # 3 tokens already home
        game.positions[0][0] = 50  # Last token close to home
        
        with patch('random.randint', return_value=5):
            next_state, reward, done, info = game.step(0)  # Move token 0
        
        assert game.home_tokens[0] == 4
        assert done
        assert game.winner == 0
        assert reward == 110  # 10 for reaching home + 100 for winning


class TestQLearningAgent:
    """Test cases for the Q-learning agent"""
    
    def test_agent_initialization(self):
        """Test that the agent initializes correctly"""
        agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.9
        assert agent.epsilon == 0.1
        assert len(agent.q_table) == 0
    
    def test_get_set_q_value(self):
        """Test getting and setting Q-values"""
        agent = QLearningAgent()
        state = (10, 20, 5)
        action = 1
        
        # Test default value
        assert agent.get_q_value(state, action) == 0.0
        
        # Test setting value
        agent.set_q_value(state, action, 5.5)
        assert agent.get_q_value(state, action) == 5.5
    
    def test_choose_action_epsilon_greedy(self):
        """Test epsilon-greedy action selection"""
        agent = QLearningAgent(epsilon=0.0)  # No exploration
        state = (10, 20, 5)
        
        # Set Q-values
        agent.set_q_value(state, 0, 1.0)  # Hold
        agent.set_q_value(state, 1, 2.0)  # Roll (better)
        
        # Should choose roll (action 1) as it has higher Q-value
        action = agent.choose_action(state)
        assert action == 1
    
    def test_update_q_value(self):
        """Test Q-value update"""
        agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9)
        state = (10, 20, 5)
        action = 1
        reward = 10
        next_state = (10, 20, 15)
        
        # Set some Q-values
        agent.set_q_value(state, action, 0.0)
        agent.set_q_value(next_state, 0, 5.0)
        agent.set_q_value(next_state, 1, 3.0)
        
        # Update Q-value
        agent.update(state, action, reward, next_state)
        
        # Check that Q-value was updated
        new_q_value = agent.get_q_value(state, action)
        expected_q_value = 0.0 + 0.1 * (10 + 0.9 * 5.0 - 0.0)
        assert abs(new_q_value - expected_q_value) < 1e-6
    
    def test_save_load_q_table(self):
        """Test saving and loading Q-table"""
        agent = QLearningAgent()
        state = (10, 20, 5)
        agent.set_q_value(state, 0, 1.5)
        agent.set_q_value(state, 1, 2.5)
        
        # Save Q-table
        import tempfile
        import os
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        temp_file.close()
        
        try:
            agent.save_q_table(temp_file.name)
            
            # Create new agent and load Q-table
            new_agent = QLearningAgent()
            new_agent.load_q_table(temp_file.name)
            
            # Check that Q-values are preserved
            assert new_agent.get_q_value(state, 0) == 1.5
            assert new_agent.get_q_value(state, 1) == 2.5
        finally:
            # Clean up
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def test_decay_epsilon(self):
        """Test epsilon decay"""
        agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01)
        
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min


class TestDQNAgent:
    """Test cases for the DQN agent"""
    
    def test_network_initialization(self):
        """Test that the DQN network initializes correctly"""
        network = DQNNetwork(input_size=3, output_size=2, hidden_sizes=[4, 3])
        assert network.network is not None
    
    def test_network_forward(self):
        """Test forward pass through the network"""
        network = DQNNetwork(input_size=3, output_size=2)
        import torch

        x = torch.FloatTensor([1.0, 2.0, 3.0])
        output = network(x)

        assert output.shape == (2,)  # Single output without batch dimension
        assert isinstance(output, torch.Tensor)
    
    def test_agent_initialization(self):
        """Test that the DQN agent initializes correctly"""
        agent = DQNAgent(state_size=3, action_size=2)
        assert agent.state_size == 3
        assert agent.action_size == 2
        assert agent.q_network is not None
        assert agent.target_network is not None
        assert len(agent.replay_buffer) == 0
    
    def test_choose_action(self):
        """Test action selection"""
        agent = DQNAgent(state_size=3, action_size=2, epsilon=0.0)  # No exploration
        state = (1.0, 2.0, 3.0)
        
        action = agent.choose_action(state)
        assert action in [0, 1]
    
    def test_store_experience(self):
        """Test storing experiences in replay buffer"""
        agent = DQNAgent(state_size=3, action_size=2)
        state = (1.0, 2.0, 3.0)
        action = 1
        reward = 10.0
        next_state = (2.0, 3.0, 4.0)
        done = False
        
        agent.store_experience(state, action, reward, next_state, done)
        
        assert len(agent.replay_buffer) == 1
        experience = agent.replay_buffer[0]
        assert experience[0] == state
        assert experience[1] == action
        assert experience[2] == reward
        assert experience[3] == next_state
        assert experience[4] == done
    
    def test_save_load_model(self):
        """Test saving and loading the model"""
        agent = DQNAgent(state_size=3, action_size=2)
        
        # Save model
        import tempfile
        import os
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
        temp_file.close()
        
        try:
            agent.save_model(temp_file.name)
            
            # Create new agent and load model
            new_agent = DQNAgent(state_size=3, action_size=2)
            new_agent.load_model(temp_file.name)
            
            # Check that epsilon is preserved
            assert new_agent.epsilon == agent.epsilon
        finally:
            # Clean up
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


class TestEvaluation:
    """Test cases for evaluation functions"""
    
    def test_evaluate_agent(self):
        """Test basic agent evaluation"""
        agent = QLearningAgent()
        
        win_rate, avg_score = evaluate_agent(agent, episodes=10, opponent_type="random")
        
        assert 0 <= win_rate <= 1
        assert avg_score >= 0
        assert isinstance(win_rate, float)
        assert isinstance(avg_score, float)
    
    def test_evaluate_agent_detailed(self):
        """Test detailed agent evaluation"""
        agent = QLearningAgent()
        
        stats = evaluate_agent_detailed(agent, episodes=10, opponent_type="random")
        
        assert "win_rate" in stats
        assert "loss_rate" in stats
        assert "avg_score" in stats
        assert "hold_rate" in stats
        assert "roll_rate" in stats
        assert 0 <= stats["win_rate"] <= 1
        assert 0 <= stats["loss_rate"] <= 1
        assert stats["avg_score"] >= 0


class TestOpponents:
    """Test cases for opponent implementations"""
    
    def test_random_opponent(self):
        """Test random opponent"""
        opponent = RandomOpponent(hold_threshold=20)
        state = (10, 20, 15)
        
        action = opponent.choose_action(state)
        assert action in [0, 1]
    
    def test_strategic_opponent(self):
        """Test strategic opponent"""
        opponent = StrategicOpponent()
        state = (10, 20, 25)
        
        action = opponent.choose_action(state)
        assert action in [0, 1]
    
    def test_random_ludo_opponent(self):
        """Test random Ludo opponent"""
        opponent = RandomLudoOpponent()
        state = tuple(range(22))  # Mock state
        valid_actions = [0, 1, 2, 3, -1]
        
        action = opponent.choose_action(state, valid_actions)
        assert action in valid_actions


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 