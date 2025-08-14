"""
Federated Learning - Streamlit Application
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys

# Set page config
st.set_page_config(
    page_title="Federated Learning",
    page_icon="🤖",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        color: #1e88e5;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .metric-card {
        background-color: #f5f9ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #1976d2;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .progress-bar {
        height: 10px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        background-color: #1976d2;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    .participant-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Simulated federated learning functions
def simulate_federated_training(epochs=5, num_participants=3):
    """Simulate federated training progress"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize metrics
    metrics = {
        'round': [],
        'accuracy': [],
        'loss': [],
        'participant_metrics': {i: {'accuracy': [], 'loss': []} for i in range(num_participants)}
    }
    
    for epoch in range(epochs):
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Training Round {epoch + 1}/{epochs} - In Progress...')
        
        # Simulate training time
        time.sleep(1)
        
        # Generate random metrics for this round
        base_accuracy = 0.6 + 0.3 * (epoch / epochs) + np.random.normal(0, 0.05)
        base_loss = 1.0 - 0.8 * (epoch / epochs) + np.random.normal(0, 0.05)
        
        metrics['round'].append(epoch + 1)
        metrics['accuracy'].append(min(0.99, max(0.6, base_accuracy)))
        metrics['loss'].append(max(0.1, min(1.0, base_loss)))
        
        # Generate participant metrics
        for p in range(num_participants):
            participant_acc = base_accuracy + np.random.normal(0, 0.05)
            participant_loss = base_loss + np.random.normal(0, 0.05)
            metrics['participant_metrics'][p]['accuracy'].append(min(0.99, max(0.5, participant_acc)))
            metrics['participant_metrics'][p]['loss'].append(max(0.1, min(1.0, participant_loss)))
    
    status_text.text('Training Complete!')
    return metrics

def plot_training_metrics(metrics):
    """Plot training metrics using Plotly"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Accuracy', 'Model Loss'))
    
    # Add global metrics
    fig.add_trace(
        go.Scatter(
            x=metrics['round'], 
            y=metrics['accuracy'],
            name='Global Model',
            line=dict(color='#1976d2', width=3)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=metrics['round'], 
            y=metrics['loss'],
            name='Global Model',
            line=dict(color='#1976d2', width=3)
        ),
        row=1, col=2
    )
    
    # Add participant metrics
    for p_id, p_metrics in metrics['participant_metrics'].items():
        fig.add_trace(
            go.Scatter(
                x=metrics['round'],
                y=p_metrics['accuracy'],
                name=f'Participant {p_id+1}',
                line=dict(dash='dash', width=1),
                opacity=0.6,
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=metrics['round'],
                y=p_metrics['loss'],
                name=f'Participant {p_id+1}',
                line=dict(dash='dash', width=1),
                opacity=0.6,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Loss", range=[0, 1.1], row=1, col=2)
    
    # Update x-axes
    fig.update_xaxes(title_text="Training Round", row=1, col=1)
    fig.update_xaxes(title_text="Training Round", row=1, col=2)
    
    return fig

def show_participant_info(num_participants):
    """Display participant information cards"""
    st.subheader("Federated Learning Participants")
    
    cols = st.columns(3)
    for i in range(num_participants):
        with cols[i % 3]:
            with st.expander(f"Participant {i+1}", expanded=True):
                st.metric("Data Samples", f"{np.random.randint(100, 1000):,}")
                st.metric("Local Epochs", np.random.choice([1, 2, 3]))
                st.metric("Last Accuracy", f"{np.random.uniform(0.7, 0.95):.2%}")
                st.metric("Last Loss", f"{np.random.uniform(0.1, 0.5):.4f}")

def main():
    """Main function to run the Streamlit app"""
    st.markdown('<h1 class="main-header">🤖 Federated Learning Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #424242;'>
            Train machine learning models across decentralized devices while keeping data localized.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Federated Learning Configuration")
    
    # Simulation parameters
    num_participants = st.sidebar.slider(
        "Number of Participants", 
        min_value=2, 
        max_value=10, 
        value=3,
        help="Number of devices/participants in the federated learning network"
    )
    
    num_rounds = st.sidebar.slider(
        "Number of Training Rounds", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="Number of federated learning rounds"
    )
    
    model_type = st.sidebar.selectbox(
        "Model Architecture",
        ["CNN", "ResNet", "LSTM", "Transformer"],
        index=0,
        help="Neural network architecture to use for training"
    )
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Participants", num_participants)
    
    with col2:
        st.metric("Training Rounds", num_rounds)
    
    with col3:
        st.metric("Model Architecture", model_type)
    
    # Start training button
    if st.button("🚀 Start Federated Training", use_container_width=True):
        with st.spinner("Initializing federated learning..."):
            # Simulate federated training
            metrics = simulate_federated_training(
                epochs=num_rounds,
                num_participants=num_participants
            )
            
            # Display results
            st.success("Federated training completed successfully!")
            
            # Show metrics
            st.subheader("Training Metrics")
            fig = plot_training_metrics(metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show participant information
            show_participant_info(num_participants)
    
    # Add some information cards
    st.markdown("---")
    st.subheader("About Federated Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🔒 Privacy Benefits", expanded=True):
            st.markdown("""
            - Data remains on local devices
            - Only model updates are shared
            - Reduces privacy risks
            - Complies with data protection regulations
            """)
        
        with st.expander("⚡ Performance", expanded=True):
            st.markdown("""
            - Faster training with parallel processing
            - Reduced network bandwidth usage
            - Works with intermittent connectivity
            - Efficient for edge devices
            """)
    
    with col2:
        with st.expander("🔍 Use Cases", expanded=True):
            st.markdown("""
            - Healthcare data analysis
            - Mobile keyboard predictions
            - Fraud detection in banking
            - IoT device learning
            - Smart home automation
            """)
        
        with st.expander("📊 Technical Details", expanded=True):
            st.markdown("""
            - Uses secure aggregation
            - Supports various ML frameworks
            - Handles non-IID data
            - Implements differential privacy
            """)
    
    # Add a footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;'>
        <p>This is a simulation of federated learning. In a real-world scenario, the training would happen on actual devices.</p>
        <p>For demonstration purposes only. Performance metrics are simulated.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
