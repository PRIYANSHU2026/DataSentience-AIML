import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.pig_game import PigGame, RandomOpponent, StrategicOpponent
from typing import Tuple, Dict, Any
import numpy as np


def evaluate_agent(agent, episodes: int = 1000, opponent_type: str = "random",
                  target_score: int = 100) -> Tuple[float, float]:
    """
    Evaluate an agent against an opponent
    
    Args:
        agent: The agent to evaluate
        episodes: Number of evaluation episodes
        opponent_type: Type of opponent ("random" or "strategic")
        target_score: Target score to win the game
        
    Returns:
        win_rate: Win rate of the agent
        avg_score: Average score of the agent
    """
    
    game = PigGame(target_score=target_score)
    
    # Choose opponent
    if opponent_type == "random":
        opponent = RandomOpponent()
    elif opponent_type == "strategic":
        opponent = StrategicOpponent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    wins = 0
    total_scores = []
    
    for _ in range(episodes):
        state, info = game.reset()
        total_reward = 0
        
        # Play one episode
        while not game.done:
            # Agent's turn
            if game.player_turn:
                action = agent.choose_action(state, epsilon=0)  # No exploration
                next_state, reward, done, info = game.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            else:
                # Opponent's turn
                opponent_action = opponent.choose_action(state)
                next_state, reward, done, info = game.step(opponent_action)
                state = next_state
                
                if done:
                    break
        
        # Check if agent won
        if game.winner == "player":
            wins += 1
        
        total_scores.append(game.player_score)
    
    win_rate = wins / episodes
    avg_score = np.mean(total_scores)
    
    return win_rate, avg_score


def evaluate_agent_detailed(agent, episodes: int = 1000, opponent_type: str = "random",
                           target_score: int = 100) -> Dict[str, Any]:
    """
    Evaluate an agent with detailed statistics
    
    Args:
        agent: The agent to evaluate
        episodes: Number of evaluation episodes
        opponent_type: Type of opponent ("random" or "strategic")
        target_score: Target score to win the game
        
    Returns:
        stats: Dictionary with detailed evaluation statistics
    """
    
    game = PigGame(target_score=target_score)
    
    # Choose opponent
    if opponent_type == "random":
        opponent = RandomOpponent()
    elif opponent_type == "strategic":
        opponent = StrategicOpponent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    wins = 0
    losses = 0
    total_scores = []
    opponent_scores = []
    game_lengths = []
    agent_actions = {"hold": 0, "roll": 0}
    
    for _ in range(episodes):
        state, info = game.reset()
        total_reward = 0
        steps = 0
        
        # Play one episode
        while not game.done:
            # Agent's turn
            if game.player_turn:
                action = agent.choose_action(state, epsilon=0)  # No exploration
                
                # Track actions
                if action == 0:
                    agent_actions["hold"] += 1
                else:
                    agent_actions["roll"] += 1
                
                next_state, reward, done, info = game.step(action)
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            else:
                # Opponent's turn
                opponent_action = opponent.choose_action(state)
                next_state, reward, done, info = game.step(opponent_action)
                state = next_state
                steps += 1
                
                if done:
                    break
        
        # Record game statistics
        if game.winner == "player":
            wins += 1
        elif game.winner == "opponent":
            losses += 1
        
        total_scores.append(game.player_score)
        opponent_scores.append(game.opponent_score)
        game_lengths.append(steps)
    
    # Calculate statistics
    win_rate = wins / episodes
    loss_rate = losses / episodes
    draw_rate = 1 - win_rate - loss_rate
    avg_score = np.mean(total_scores)
    avg_opponent_score = np.mean(opponent_scores)
    avg_game_length = np.mean(game_lengths)
    
    # Action distribution
    total_actions = agent_actions["hold"] + agent_actions["roll"]
    if total_actions > 0:
        hold_rate = agent_actions["hold"] / total_actions
        roll_rate = agent_actions["roll"] / total_actions
    else:
        hold_rate = roll_rate = 0
    
    stats = {
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "avg_score": avg_score,
        "avg_opponent_score": avg_opponent_score,
        "avg_game_length": avg_game_length,
        "hold_rate": hold_rate,
        "roll_rate": roll_rate,
        "total_episodes": episodes,
        "opponent_type": opponent_type,
        "target_score": target_score
    }
    
    return stats


def compare_agents(agents: Dict[str, Any], episodes: int = 1000, 
                  opponent_type: str = "random", target_score: int = 100) -> Dict[str, Any]:
    """
    Compare multiple agents against the same opponent
    
    Args:
        agents: Dictionary mapping agent names to agent objects
        episodes: Number of evaluation episodes per agent
        opponent_type: Type of opponent
        target_score: Target score to win the game
        
    Returns:
        comparison: Dictionary with comparison statistics
    """
    
    comparison = {}
    
    for agent_name, agent in agents.items():
        print(f"Evaluating {agent_name}...")
        stats = evaluate_agent_detailed(agent, episodes, opponent_type, target_score)
        comparison[agent_name] = stats
    
    return comparison


def evaluate_agent_progression(agent, episodes: int = 1000, opponent_type: str = "random",
                             target_score: int = 100, eval_interval: int = 100) -> Dict[str, list]:
    """
    Evaluate an agent's progression over time
    
    Args:
        agent: The agent to evaluate
        episodes: Total number of episodes to evaluate
        opponent_type: Type of opponent
        target_score: Target score to win the game
        eval_interval: Interval between evaluations
        
    Returns:
        progression: Dictionary with progression statistics
    """
    
    progression = {
        "episodes": [],
        "win_rates": [],
        "avg_scores": [],
        "hold_rates": [],
        "roll_rates": []
    }
    
    for i in range(0, episodes, eval_interval):
        # Create a copy of the agent for evaluation
        if hasattr(agent, 'load_q_table'):
            # Q-learning agent
            import tempfile
            import pickle
            
            # Save current state
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            agent.save_q_table(temp_file.name)
            
            # Create new agent and load state
            from agents.q_learning_agent import QLearningAgent
            eval_agent = QLearningAgent()
            eval_agent.load_q_table(temp_file.name)
            
            # Clean up
            import os
            os.unlink(temp_file.name)
        else:
            # DQN agent
            import tempfile
            import torch
            
            # Save current state
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
            agent.save_model(temp_file.name)
            
            # Create new agent and load state
            from agents.dqn_agent import DQNAgent
            eval_agent = DQNAgent(agent.state_size, agent.action_size)
            eval_agent.load_model(temp_file.name)
            
            # Clean up
            import os
            os.unlink(temp_file.name)
        
        # Evaluate
        stats = evaluate_agent_detailed(eval_agent, eval_interval, opponent_type, target_score)
        
        progression["episodes"].append(i + eval_interval)
        progression["win_rates"].append(stats["win_rate"])
        progression["avg_scores"].append(stats["avg_score"])
        progression["hold_rates"].append(stats["hold_rate"])
        progression["roll_rates"].append(stats["roll_rate"])
    
    return progression


def print_evaluation_results(stats: Dict[str, Any]):
    """
    Print evaluation results in a formatted way
    
    Args:
        stats: Evaluation statistics dictionary
    """
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Opponent Type: {stats['opponent_type']}")
    print(f"Target Score: {stats['target_score']}")
    print(f"Episodes: {stats['total_episodes']}")
    print("-"*50)
    print(f"Win Rate: {stats['win_rate']:.2%}")
    print(f"Loss Rate: {stats['loss_rate']:.2%}")
    print(f"Draw Rate: {stats['draw_rate']:.2%}")
    print("-"*50)
    print(f"Average Score: {stats['avg_score']:.2f}")
    print(f"Average Opponent Score: {stats['avg_opponent_score']:.2f}")
    print(f"Average Game Length: {stats['avg_game_length']:.1f} steps")
    print("-"*50)
    print(f"Hold Rate: {stats['hold_rate']:.2%}")
    print(f"Roll Rate: {stats['roll_rate']:.2%}")
    print("="*50)


def print_comparison_results(comparison: Dict[str, Dict[str, Any]]):
    """
    Print comparison results in a formatted way
    
    Args:
        comparison: Comparison statistics dictionary
    """
    
    print("\n" + "="*70)
    print("AGENT COMPARISON")
    print("="*70)
    
    # Print header
    print(f"{'Agent':<20} {'Win Rate':<12} {'Avg Score':<12} {'Hold Rate':<12} {'Roll Rate':<12}")
    print("-"*70)
    
    # Print results for each agent
    for agent_name, stats in comparison.items():
        print(f"{agent_name:<20} {stats['win_rate']:<12.2%} {stats['avg_score']:<12.2f} "
              f"{stats['hold_rate']:<12.2%} {stats['roll_rate']:<12.2%}")
    
    print("="*70)


if __name__ == "__main__":
    # Example usage
    from agents.q_learning_agent import QLearningAgent
    
    # Create a simple agent for testing
    agent = QLearningAgent()
    
    # Evaluate the agent
    stats = evaluate_agent_detailed(agent, episodes=100, opponent_type="random")
    print_evaluation_results(stats) 