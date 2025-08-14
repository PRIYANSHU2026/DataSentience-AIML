import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_agent import DQNAgent
from environments.pig_game import PigGame, RandomOpponent, StrategicOpponent
from evaluation.evaluate_agent import evaluate_agent
from utils.plotting import plot_training_stats
from tqdm import tqdm
import numpy as np
from typing import Dict, Any


def train_dqn_agent(episodes: int = 10000, learning_rate: float = 0.001,
                   discount_factor: float = 0.9, epsilon: float = 1.0,
                   epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                   eval_freq: int = 1000, opponent_type: str = "random",
                   target_score: int = 100, replay_buffer_size: int = 10000,
                   batch_size: int = 32, target_update_freq: int = 100,
                   hidden_sizes: list = None) -> DQNAgent:
    """
    Train a DQN agent on the Pig game
    
    Args:
        episodes: Number of training episodes
        learning_rate: Learning rate for the neural network
        discount_factor: Discount factor for future rewards
        epsilon: Initial exploration rate
        epsilon_decay: Rate at which epsilon decays
        epsilon_min: Minimum epsilon value
        eval_freq: Frequency of evaluation episodes
        opponent_type: Type of opponent ("random" or "strategic")
        target_score: Target score to win the game
        replay_buffer_size: Size of experience replay buffer
        batch_size: Batch size for training
        target_update_freq: Frequency of target network updates
        hidden_sizes: List of hidden layer sizes for the neural network
        
    Returns:
        trained_agent: The trained DQN agent
    """
    
    # State size for Pig game: (player_score, opponent_score, turn_total)
    state_size = 3
    action_size = 2  # hold, roll
    
    if hidden_sizes is None:
        hidden_sizes = [128, 64]
    
    # Initialize agent and environment
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        hidden_sizes=hidden_sizes
    )
    
    game = PigGame(target_score=target_score)
    
    # Choose opponent
    if opponent_type == "random":
        opponent = RandomOpponent()
    elif opponent_type == "strategic":
        opponent = StrategicOpponent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    # Training statistics
    episode_rewards = []
    win_rates = []
    avg_scores = []
    
    print(f"Training DQN agent for {episodes} episodes...")
    print(f"Opponent: {opponent_type}")
    print(f"Target score: {target_score}")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Hidden layers: {hidden_sizes}")
    print(f"Replay buffer size: {replay_buffer_size}")
    print(f"Batch size: {batch_size}")
    print("-" * 50)
    
    # Training loop
    for episode in tqdm(range(episodes), desc="Training"):
        state, info = game.reset()
        total_reward = 0
        steps = 0
        
        # Play one episode
        while not game.done:
            # Agent's turn
            if game.player_turn:
                action = agent.choose_action(state)
                next_state, reward, done, info = game.step(action)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train the network
                agent.train()
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            else:
                # Opponent's turn
                opponent_action = opponent.choose_action(state)
                next_state, reward, done, info = game.step(opponent_action)
                
                # Store experience (negative reward for opponent's actions)
                agent.store_experience(state, opponent_action, -reward, next_state, done)
                
                # Train the network
                agent.train()
                
                state = next_state
                steps += 1
                
                if done:
                    break
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        agent.add_episode_reward(total_reward)
        
        # Evaluate periodically
        if (episode + 1) % eval_freq == 0:
            win_rate, avg_score = evaluate_agent(agent, episodes=100, opponent_type=opponent_type)
            win_rates.append(win_rate)
            avg_scores.append(avg_score)
            agent.add_win_rate(win_rate)
            
            print(f"Episode {episode + 1}: Win Rate = {win_rate:.2%}, Avg Score = {avg_score:.2f}, Epsilon = {agent.epsilon:.3f}")
            print(f"  Replay buffer size: {agent.get_replay_buffer_size()}")
            if agent.losses:
                print(f"  Average loss: {np.mean(agent.losses[-100:]):.4f}")
        
        # Decay epsilon
        agent.decay_epsilon()
    
    # Final evaluation
    final_win_rate, final_avg_score = evaluate_agent(agent, episodes=1000, opponent_type=opponent_type)
    print(f"\nFinal Results:")
    print(f"Win Rate: {final_win_rate:.2%}")
    print(f"Average Score: {final_avg_score:.2f}")
    print(f"Replay buffer size: {agent.get_replay_buffer_size()}")
    if agent.losses:
        print(f"Final average loss: {np.mean(agent.losses[-100:]):.4f}")
    
    return agent


def train_dqn_agent_ludo(episodes: int = 10000, learning_rate: float = 0.001,
                         discount_factor: float = 0.9, epsilon: float = 1.0,
                         epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                         eval_freq: int = 1000, opponent_type: str = "random",
                         replay_buffer_size: int = 10000, batch_size: int = 32,
                         target_update_freq: int = 100, hidden_sizes: list = None) -> DQNAgent:
    """
    Train a DQN agent on the Ludo game
    
    Args:
        episodes: Number of training episodes
        learning_rate: Learning rate for the neural network
        discount_factor: Discount factor for future rewards
        epsilon: Initial exploration rate
        epsilon_decay: Rate at which epsilon decays
        epsilon_min: Minimum epsilon value
        eval_freq: Frequency of evaluation episodes
        opponent_type: Type of opponent ("random" or "strategic")
        replay_buffer_size: Size of experience replay buffer
        batch_size: Batch size for training
        target_update_freq: Frequency of target network updates
        hidden_sizes: List of hidden layer sizes for the neural network
        
    Returns:
        trained_agent: The trained DQN agent
    """
    
    from environments.ludo_game import LudoGame, RandomLudoOpponent, StrategicLudoOpponent
    
    # State size for Ludo game: 16 positions + 4 home tokens + current player + die roll
    state_size = 22
    action_size = 5  # 4 tokens + no move
    
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]
    
    # Initialize agent and environment
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        hidden_sizes=hidden_sizes
    )
    
    game = LudoGame()
    
    # Choose opponent
    if opponent_type == "random":
        opponent = RandomLudoOpponent()
    elif opponent_type == "strategic":
        opponent = StrategicLudoOpponent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    # Training statistics
    episode_rewards = []
    win_rates = []
    avg_scores = []
    
    print(f"Training DQN agent on Ludo for {episodes} episodes...")
    print(f"Opponent: {opponent_type}")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Hidden layers: {hidden_sizes}")
    print(f"Replay buffer size: {replay_buffer_size}")
    print(f"Batch size: {batch_size}")
    print("-" * 50)
    
    # Training loop
    for episode in tqdm(range(episodes), desc="Training"):
        state, info = game.reset()
        total_reward = 0
        steps = 0
        
        # Play one episode
        while not game.done:
            # Get valid actions
            valid_actions = game.get_valid_actions()
            
            # Agent's turn
            if game.current_player == 0:  # Assume agent is player 0
                action = agent.choose_action(state, valid_actions)
                next_state, reward, done, info = game.step(action)
                
                # Store experience
                next_valid_actions = game.get_valid_actions()
                agent.store_experience(state, action, reward, next_state, done, next_valid_actions)
                
                # Train the network
                agent.train()
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            else:
                # Opponent's turn
                opponent_action = opponent.choose_action(state, valid_actions)
                next_state, reward, done, info = game.step(opponent_action)
                
                # Store experience (negative reward for opponent's actions)
                next_valid_actions = game.get_valid_actions()
                agent.store_experience(state, opponent_action, -reward, next_state, done, next_valid_actions)
                
                # Train the network
                agent.train()
                
                state = next_state
                steps += 1
                
                if done:
                    break
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        agent.add_episode_reward(total_reward)
        
        # Evaluate periodically
        if (episode + 1) % eval_freq == 0:
            # For Ludo, we need a different evaluation function
            win_rate = evaluate_ludo_agent(agent, episodes=100, opponent_type=opponent_type)
            win_rates.append(win_rate)
            agent.add_win_rate(win_rate)
            
            print(f"Episode {episode + 1}: Win Rate = {win_rate:.2%}, Epsilon = {agent.epsilon:.3f}")
            print(f"  Replay buffer size: {agent.get_replay_buffer_size()}")
            if agent.losses:
                print(f"  Average loss: {np.mean(agent.losses[-100:]):.4f}")
        
        # Decay epsilon
        agent.decay_epsilon()
    
    # Final evaluation
    final_win_rate = evaluate_ludo_agent(agent, episodes=1000, opponent_type=opponent_type)
    print(f"\nFinal Results:")
    print(f"Win Rate: {final_win_rate:.2%}")
    print(f"Replay buffer size: {agent.get_replay_buffer_size()}")
    if agent.losses:
        print(f"Final average loss: {np.mean(agent.losses[-100:]):.4f}")
    
    return agent


def evaluate_ludo_agent(agent: DQNAgent, episodes: int = 100, opponent_type: str = "random") -> float:
    """
    Evaluate a DQN agent on the Ludo game
    
    Args:
        agent: The agent to evaluate
        episodes: Number of evaluation episodes
        opponent_type: Type of opponent
        
    Returns:
        win_rate: Win rate of the agent
    """
    from environments.ludo_game import LudoGame, RandomLudoOpponent, StrategicLudoOpponent
    
    game = LudoGame()
    
    if opponent_type == "random":
        opponent = RandomLudoOpponent()
    elif opponent_type == "strategic":
        opponent = StrategicLudoOpponent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    wins = 0
    
    for _ in range(episodes):
        state, info = game.reset()
        
        while not game.done:
            valid_actions = game.get_valid_actions()
            
            if game.current_player == 0:  # Agent's turn
                action = agent.choose_action(state, valid_actions, epsilon=0)  # No exploration
                next_state, reward, done, info = game.step(action)
                state = next_state
            else:  # Opponent's turn
                opponent_action = opponent.choose_action(state, valid_actions)
                next_state, reward, done, info = game.step(opponent_action)
                state = next_state
        
        # Check if agent won
        if game.winner == 0:
            wins += 1
    
    return wins / episodes


if __name__ == "__main__":
    # Example usage for Pig game
    agent = train_dqn_agent(
        episodes=5000,
        learning_rate=0.001,
        discount_factor=0.9,
        epsilon=1.0,
        opponent_type="random"
    )
    
    # Save trained agent
    agent.save_model("trained_dqn_agent.pth")
    
    # Plot training statistics
    stats = agent.get_training_stats()
    plot_training_stats(stats, "DQN Training Statistics") 