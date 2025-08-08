#!/usr/bin/env python3
"""
AI DiceMaster: Main Demo Script

This script demonstrates the AI DiceMaster project by training agents
and showing their performance on dice games.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_q_learning import train_q_learning_agent
from training.train_dqn import train_dqn_agent
from evaluation.evaluate_agent import evaluate_agent_detailed, print_evaluation_results
from utils.plotting import plot_training_stats, plot_agent_comparison
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from environments.pig_game import PigGame, RandomOpponent, StrategicOpponent


def demo_pig_game():
    """Demonstrate the Pig game environment"""
    print("=" * 60)
    print("🎲 PIG GAME DEMONSTRATION")
    print("=" * 60)
    
    game = PigGame(target_score=20)  # Shorter game for demo
    state, info = game.reset()
    
    print("Initial game state:")
    game.render()
    
    # Play a few moves
    for i in range(5):
        if game.done:
            break
            
        if game.player_turn:
            action = 1  # Roll
            print(f"\nPlayer's turn - Rolling...")
        else:
            action = 0  # Hold
            print(f"\nOpponent's turn - Holding...")
            
        state, reward, done, info = game.step(action)
        game.render()
    
    print("\nGame demonstration completed!")


def demo_q_learning_training():
    """Demonstrate Q-learning training"""
    print("\n" + "=" * 60)
    print("🤖 Q-LEARNING TRAINING DEMONSTRATION")
    print("=" * 60)
    
    print("Training Q-learning agent...")
    print("This will take a few moments...")
    
    # Train a Q-learning agent
    agent = train_q_learning_agent(
        episodes=1000,  # Reduced for demo
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        opponent_type="random",
        eval_freq=200
    )
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    stats = evaluate_agent_detailed(agent, episodes=100, opponent_type="random")
    print_evaluation_results(stats)
    
    # Plot training statistics
    print("\nGenerating training plots...")
    training_stats = agent.get_training_stats()
    plot_training_stats(training_stats, "Q-Learning Training Statistics")
    
    return agent


def demo_dqn_training():
    """Demonstrate DQN training"""
    print("\n" + "=" * 60)
    print("🧠 DEEP Q-NETWORK TRAINING DEMONSTRATION")
    print("=" * 60)
    
    print("Training DQN agent...")
    print("This will take a few moments...")
    
    # Train a DQN agent
    agent = train_dqn_agent(
        episodes=1000,  # Reduced for demo
        learning_rate=0.001,
        discount_factor=0.9,
        epsilon=1.0,
        opponent_type="random",
        eval_freq=200,
        replay_buffer_size=5000,
        batch_size=16
    )
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    stats = evaluate_agent_detailed(agent, episodes=100, opponent_type="random")
    print_evaluation_results(stats)
    
    # Plot training statistics
    print("\nGenerating training plots...")
    training_stats = agent.get_training_stats()
    plot_training_stats(training_stats, "DQN Training Statistics")
    
    return agent


def demo_agent_comparison():
    """Demonstrate agent comparison"""
    print("\n" + "=" * 60)
    print("📊 AGENT COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Create agents
    agents = {
        "Q-Learning": QLearningAgent(),
        "DQN": DQNAgent(state_size=3, action_size=2)
    }
    
    # Train agents briefly
    print("Training agents for comparison...")
    
    # Quick training for Q-learning
    q_agent = train_q_learning_agent(
        episodes=500,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        opponent_type="random",
        eval_freq=100
    )
    
    # Quick training for DQN
    dqn_agent = train_dqn_agent(
        episodes=500,
        learning_rate=0.001,
        discount_factor=0.9,
        epsilon=1.0,
        opponent_type="random",
        eval_freq=100,
        replay_buffer_size=2000,
        batch_size=16
    )
    
    # Compare agents
    trained_agents = {
        "Q-Learning": q_agent,
        "DQN": dqn_agent
    }
    
    print("\nComparing trained agents...")
    comparison = {}
    
    for name, agent in trained_agents.items():
        print(f"Evaluating {name}...")
        stats = evaluate_agent_detailed(agent, episodes=100, opponent_type="random")
        comparison[name] = stats
    
    # Print comparison results
    from evaluation.evaluate_agent import print_comparison_results
    print_comparison_results(comparison)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_agent_comparison(comparison, "Agent Comparison")
    
    return trained_agents


def demo_play_against_agent():
    """Demonstrate playing against a trained agent"""
    print("\n" + "=" * 60)
    print("🎮 PLAY AGAINST TRAINED AGENT")
    print("=" * 60)
    
    # Train a simple agent
    print("Training an agent to play against...")
    agent = train_q_learning_agent(
        episodes=500,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        opponent_type="random",
        eval_freq=100
    )
    
    # Play a game against the agent
    game = PigGame(target_score=30)
    state, info = game.reset()
    
    print("\nPlaying a game against the trained agent:")
    print("You are the opponent (O), the agent is the player (P)")
    print("-" * 50)
    
    while not game.done:
        if game.player_turn:
            # Agent's turn
            action = agent.choose_action(state, epsilon=0)  # No exploration
            action_name = "HOLD" if action == 0 else "ROLL"
            print(f"Agent's turn: {action_name}")
            state, reward, done, info = game.step(action)
        else:
            # Human's turn (simulated)
            opponent = RandomOpponent()
            action = opponent.choose_action(state)
            action_name = "HOLD" if action == 0 else "ROLL"
            print(f"Your turn: {action_name}")
            state, reward, done, info = game.step(action)
        
        game.render()
    
    print(f"\nGame Over! Winner: {game.winner}")


def main():
    """Main demonstration function"""
    print("🎲 AI DICEMASTER - REINFORCEMENT LEARNING FOR DICE GAMES")
    print("=" * 70)
    print("This demonstration shows how AI agents learn to play dice games")
    print("using Q-learning and Deep Q-Networks (DQN).")
    print()
    
    try:
        # Demo 1: Pig Game Environment
        demo_pig_game()
        
        # Demo 2: Q-Learning Training
        q_agent = demo_q_learning_training()
        
        # Demo 3: DQN Training
        dqn_agent = demo_dqn_training()
        
        # Demo 4: Agent Comparison
        agents = demo_agent_comparison()
        
        # Demo 5: Play Against Agent
        demo_play_against_agent()
        
        print("\n" + "=" * 70)
        print("🎉 DEMONSTRATION COMPLETED!")
        print("=" * 70)
        print("The AI DiceMaster project has successfully demonstrated:")
        print("✅ Pig game environment with RL interface")
        print("✅ Q-learning agent training and evaluation")
        print("✅ DQN agent training and evaluation")
        print("✅ Agent comparison and visualization")
        print("✅ Interactive gameplay demonstration")
        print()
        print("You can now explore the codebase and experiment with:")
        print("- Different training parameters")
        print("- New game environments")
        print("- Advanced RL algorithms")
        print("- Custom reward functions")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check that all dependencies are installed correctly.")


if __name__ == "__main__":
    main() 