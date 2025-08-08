import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.q_learning_agent import QLearningAgent
from environments.pig_game import PigGame, RandomOpponent, StrategicOpponent
from evaluation.evaluate_agent import evaluate_agent
from utils.plotting import plot_training_stats
from tqdm import tqdm
import numpy as np
from typing import Dict, Any


def train_q_learning_agent(episodes: int = 10000, learning_rate: float = 0.1,
                          discount_factor: float = 0.9, epsilon: float = 0.1,
                          epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                          eval_freq: int = 1000, opponent_type: str = "random",
                          target_score: int = 100) -> QLearningAgent:
    """
    Train a Q-learning agent on the Pig game
    
    Args:
        episodes: Number of training episodes
        learning_rate: Q-learning learning rate
        discount_factor: Discount factor for future rewards
        epsilon: Initial exploration rate
        epsilon_decay: Rate at which epsilon decays
        epsilon_min: Minimum epsilon value
        eval_freq: Frequency of evaluation episodes
        opponent_type: Type of opponent ("random" or "strategic")
        target_score: Target score to win the game
        
    Returns:
        trained_agent: The trained Q-learning agent
    """
    
    # Initialize agent and environment
    agent = QLearningAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min
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
    
    print(f"Training Q-learning agent for {episodes} episodes...")
    print(f"Opponent: {opponent_type}")
    print(f"Target score: {target_score}")
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
                
                # Update agent
                agent.update(state, action, reward, next_state)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            else:
                # Opponent's turn
                opponent_action = opponent.choose_action(state)
                next_state, reward, done, info = game.step(opponent_action)
                
                # Negative reward for opponent's actions
                agent.update(state, opponent_action, -reward, next_state)
                
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
        
        # Decay epsilon
        agent.decay_epsilon()
    
    # Final evaluation
    final_win_rate, final_avg_score = evaluate_agent(agent, episodes=1000, opponent_type=opponent_type)
    print(f"\nFinal Results:")
    print(f"Win Rate: {final_win_rate:.2%}")
    print(f"Average Score: {final_avg_score:.2f}")
    print(f"Q-table size: {agent.get_q_table_size()}")
    
    return agent


def train_q_learning_with_experience_replay(episodes: int = 10000, learning_rate: float = 0.1,
                                          discount_factor: float = 0.9, epsilon: float = 0.1,
                                          epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                                          eval_freq: int = 1000, opponent_type: str = "random",
                                          target_score: int = 100, replay_buffer_size: int = 10000,
                                          batch_size: int = 32) -> QLearningAgent:
    """
    Train a Q-learning agent with experience replay on the Pig game
    
    Args:
        episodes: Number of training episodes
        learning_rate: Q-learning learning rate
        discount_factor: Discount factor for future rewards
        epsilon: Initial exploration rate
        epsilon_decay: Rate at which epsilon decays
        epsilon_min: Minimum epsilon value
        eval_freq: Frequency of evaluation episodes
        opponent_type: Type of opponent ("random" or "strategic")
        target_score: Target score to win the game
        replay_buffer_size: Size of experience replay buffer
        batch_size: Batch size for experience replay
        
    Returns:
        trained_agent: The trained Q-learning agent with experience replay
    """
    
    from agents.q_learning_agent import QLearningAgentWithExperienceReplay
    
    # Initialize agent and environment
    agent = QLearningAgentWithExperienceReplay(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size
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
    
    print(f"Training Q-learning agent with experience replay for {episodes} episodes...")
    print(f"Opponent: {opponent_type}")
    print(f"Target score: {target_score}")
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
                
                # Train on batch of experiences
                agent.replay()
                
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
                
                # Train on batch of experiences
                agent.replay()
                
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
        
        # Decay epsilon
        agent.decay_epsilon()
    
    # Final evaluation
    final_win_rate, final_avg_score = evaluate_agent(agent, episodes=1000, opponent_type=opponent_type)
    print(f"\nFinal Results:")
    print(f"Win Rate: {final_win_rate:.2%}")
    print(f"Average Score: {final_avg_score:.2f}")
    print(f"Q-table size: {agent.get_q_table_size()}")
    print(f"Replay buffer size: {len(agent.replay_buffer)}")
    
    return agent


if __name__ == "__main__":
    # Example usage
    agent = train_q_learning_agent(
        episodes=5000,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        opponent_type="random"
    )
    
    # Save trained agent
    agent.save_q_table("trained_q_learning_agent.pkl")
    
    # Plot training statistics
    stats = agent.get_training_stats()
    plot_training_stats(stats, "Q-Learning Training Statistics") 