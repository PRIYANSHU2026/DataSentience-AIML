import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import os


def plot_training_stats(stats: Dict[str, List[float]], title: str = "Training Statistics"):
    """
    Plot training statistics including rewards, win rates, and losses
    
    Args:
        stats: Dictionary containing training statistics
        title: Title for the plot
    """
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot episode rewards
    if 'episode_rewards' in stats and stats['episode_rewards']:
        axes[0, 0].plot(stats['episode_rewards'], alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add moving average
        if len(stats['episode_rewards']) > 100:
            window = min(100, len(stats['episode_rewards']) // 10)
            moving_avg = np.convolve(stats['episode_rewards'], 
                                    np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(stats['episode_rewards'])), 
                           moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
            axes[0, 0].legend()
    
    # Plot win rates
    if 'win_rates' in stats and stats['win_rates']:
        axes[0, 1].plot(stats['win_rates'], 'g-', linewidth=2)
        axes[0, 1].set_title('Win Rate Progression')
        axes[0, 1].set_xlabel('Evaluation Step')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
    
    # Plot training rewards (if different from episode rewards)
    if 'training_rewards' in stats and stats['training_rewards']:
        axes[1, 0].plot(stats['training_rewards'], alpha=0.7, linewidth=1)
        axes[1, 0].set_title('Training Rewards')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot losses (for DQN)
    if 'losses' in stats and stats['losses']:
        axes[1, 1].plot(stats['losses'], 'r-', alpha=0.7, linewidth=1)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add moving average for losses
        if len(stats['losses']) > 100:
            window = min(100, len(stats['losses']) // 10)
            moving_avg = np.convolve(stats['losses'], 
                                    np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(stats['losses'])), 
                           moving_avg, 'b-', linewidth=2, label=f'Moving Average (window={window})')
            axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_agent_comparison(comparison: Dict[str, Dict[str, Any]], title: str = "Agent Comparison"):
    """
    Plot comparison of multiple agents
    
    Args:
        comparison: Dictionary mapping agent names to their statistics
        title: Title for the plot
    """
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Extract data
    agent_names = list(comparison.keys())
    win_rates = [comparison[name]['win_rate'] for name in agent_names]
    avg_scores = [comparison[name]['avg_score'] for name in agent_names]
    hold_rates = [comparison[name]['hold_rate'] for name in agent_names]
    roll_rates = [comparison[name]['roll_rate'] for name in agent_names]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Win rates
    bars1 = axes[0, 0].bar(agent_names, win_rates, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Win Rates')
    axes[0, 0].set_ylabel('Win Rate')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2%}', ha='center', va='bottom')
    
    # Average scores
    bars2 = axes[0, 1].bar(agent_names, avg_scores, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Average Scores')
    axes[0, 1].set_ylabel('Average Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom')
    
    # Action distribution
    x = np.arange(len(agent_names))
    width = 0.35
    
    bars3 = axes[1, 0].bar(x - width/2, hold_rates, width, label='Hold', color='orange', alpha=0.7)
    bars4 = axes[1, 0].bar(x + width/2, roll_rates, width, label='Roll', color='red', alpha=0.7)
    
    axes[1, 0].set_title('Action Distribution')
    axes[1, 0].set_ylabel('Action Rate')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(agent_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    
    # Radar chart for overall performance
    categories = ['Win Rate', 'Avg Score', 'Hold Rate', 'Roll Rate']
    
    # Normalize scores for radar chart
    norm_win_rates = np.array(win_rates)
    norm_avg_scores = np.array(avg_scores) / max(avg_scores) if max(avg_scores) > 0 else np.array(avg_scores)
    norm_hold_rates = np.array(hold_rates)
    norm_roll_rates = np.array(roll_rates)
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    axes[1, 1].remove()
    ax_radar = fig.add_subplot(2, 2, 4, projection='polar')
    
    for i, agent_name in enumerate(agent_names):
        values = [norm_win_rates[i], norm_avg_scores[i], norm_hold_rates[i], norm_roll_rates[i]]
        values += values[:1]  # Complete the circle
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=agent_name)
        ax_radar.fill(angles, values, alpha=0.25)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Performance Radar Chart')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.show()


def plot_learning_curves(agents: Dict[str, Any], episodes: int = 1000, 
                        opponent_type: str = "random", title: str = "Learning Curves"):
    """
    Plot learning curves for multiple agents
    
    Args:
        agents: Dictionary mapping agent names to agent objects
        episodes: Number of episodes to evaluate
        opponent_type: Type of opponent for evaluation
        title: Title for the plot
    """
    
    from evaluation.evaluate_agent import evaluate_agent_progression
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for agent_name, agent in agents.items():
        print(f"Evaluating {agent_name} progression...")
        progression = evaluate_agent_progression(agent, episodes, opponent_type)
        
        # Plot win rates
        axes[0, 0].plot(progression['episodes'], progression['win_rates'], 
                        label=agent_name, linewidth=2)
        axes[0, 0].set_title('Win Rate Progression')
        axes[0, 0].set_xlabel('Episodes')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # Plot average scores
        axes[0, 1].plot(progression['episodes'], progression['avg_scores'], 
                        label=agent_name, linewidth=2)
        axes[0, 1].set_title('Average Score Progression')
        axes[0, 1].set_xlabel('Episodes')
        axes[0, 1].set_ylabel('Average Score')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot hold rates
        axes[1, 0].plot(progression['episodes'], progression['hold_rates'], 
                        label=agent_name, linewidth=2)
        axes[1, 0].set_title('Hold Rate Progression')
        axes[1, 0].set_xlabel('Episodes')
        axes[1, 0].set_ylabel('Hold Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        
        # Plot roll rates
        axes[1, 1].plot(progression['episodes'], progression['roll_rates'], 
                        label=agent_name, linewidth=2)
        axes[1, 1].set_title('Roll Rate Progression')
        axes[1, 1].set_xlabel('Episodes')
        axes[1, 1].set_ylabel('Roll Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()


def plot_game_statistics(game_stats: List[Dict[str, Any]], title: str = "Game Statistics"):
    """
    Plot statistics from individual games
    
    Args:
        game_stats: List of dictionaries containing game statistics
        title: Title for the plot
    """
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Extract data
    game_lengths = [game['length'] for game in game_stats]
    player_scores = [game['player_score'] for game in game_stats]
    opponent_scores = [game['opponent_score'] for game in game_stats]
    winners = [game['winner'] for game in game_stats]
    
    # Game length distribution
    axes[0, 0].hist(game_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Game Length Distribution')
    axes[0, 0].set_xlabel('Game Length (steps)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Score distribution
    axes[0, 1].hist(player_scores, bins=20, alpha=0.7, color='lightgreen', 
                    label='Player', edgecolor='black')
    axes[0, 1].hist(opponent_scores, bins=20, alpha=0.7, color='lightcoral', 
                    label='Opponent', edgecolor='black')
    axes[0, 1].set_title('Score Distribution')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Winner distribution
    winner_counts = {}
    for winner in winners:
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
    
    if winner_counts:
        axes[1, 0].pie(winner_counts.values(), labels=winner_counts.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Winner Distribution')
    
    # Score correlation
    axes[1, 1].scatter(player_scores, opponent_scores, alpha=0.6)
    axes[1, 1].set_title('Player vs Opponent Scores')
    axes[1, 1].set_xlabel('Player Score')
    axes[1, 1].set_ylabel('Opponent Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add trend line
    if len(player_scores) > 1:
        z = np.polyfit(player_scores, opponent_scores, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(player_scores, p(player_scores), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()


def save_plots_to_file(fig, filename: str, dpi: int = 300):
    """
    Save a matplotlib figure to file
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename
        dpi: Resolution for saving
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {filename}")


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample training statistics
    sample_stats = {
        'episode_rewards': np.random.normal(0, 10, 1000),
        'win_rates': np.linspace(0.3, 0.8, 20),
        'training_rewards': np.random.normal(0, 5, 1000),
        'losses': np.random.exponential(1, 1000)
    }
    
    # Plot the statistics
    plot_training_stats(sample_stats, "Sample Training Statistics") 