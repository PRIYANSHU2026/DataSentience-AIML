import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="Entertainment Industry Analytics",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #6A1B9A;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A148C;
        font-weight: 500;
    }
    .tool-card {
        background-color: #f5f0fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #6A1B9A;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .tool-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .tool-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4A148C;
    }
    .tool-desc {
        color: #424242;
        margin-bottom: 15px;
    }
    .stButton button {
        background-color: #6A1B9A;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 0.5em 1em;
        border: none;
    }
    .stButton button:hover {
        background-color: #4A148C;
    }
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6A1B9A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #546E7A;
        margin-top: 0.3rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #78909C;
        border-top: 1px solid #eceff1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for tool selection if not already set
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = None

# Function to create a tool card
def tool_card(title, description, icon, button_text, button_key):
    st.markdown(f'''
    <div class="tool-card">
        <p class="tool-title">{icon} {title}</p>
        <p class="tool-desc">{description}</p>
    </div>
    ''', unsafe_allow_html=True)
    return st.button(button_text, key=button_key)

# Main header
st.markdown('<div class="main-header">🎬 Entertainment Industry Analytics</div>', unsafe_allow_html=True)

# Function to go back to main dashboard
def go_back_to_main():
    st.session_state.selected_tool = None

# Display selected tool or main dashboard
if st.session_state.selected_tool:
    # Back button
    if st.button("← Back to Dashboard"):
        go_back_to_main()
    
    # Restaurant Insight AI Tool
    if st.session_state.selected_tool == "restaurant_insight":
        st.markdown('<p class="sub-header">🍽️ RestaurantInsight AI</p>', unsafe_allow_html=True)
        st.markdown("Advanced restaurant analytics and customer intelligence platform")
        
        # Sample tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Restaurant Clustering", "Sentiment Analysis", "Market Intelligence"])
        
        with tab1:
            st.markdown("### Restaurant Clustering Analysis")
            st.markdown("Segment restaurants based on cuisine, pricing, and location.")
            
            # Sample clustering visualization
            st.markdown("#### Restaurant Clusters")
            
            # Generate sample data for demonstration
            np.random.seed(42)
            cluster_data = pd.DataFrame({
                'Restaurant': [f'Restaurant {i}' for i in range(1, 101)],
                'Price': np.random.randint(100, 2000, 100),
                'Rating': np.random.uniform(3.0, 5.0, 100),
                'Cluster': np.random.choice(['Premium Fine Dining', 'Casual Dining', 'Fast Casual', 'Budget Eateries', 'Specialty Restaurants'], 100)
            })
            
            fig = px.scatter(cluster_data, x='Price', y='Rating', color='Cluster', hover_name='Restaurant',
                           title='Restaurant Clustering by Price and Rating')
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster descriptions
            st.markdown("#### Cluster Characteristics")
            clusters = {
                "Premium Fine Dining": "High-end restaurants, expensive, diverse cuisines",
                "Casual Dining": "Mid-range restaurants, popular cuisines",
                "Fast Casual": "Quick service, moderate pricing",
                "Budget Eateries": "Affordable options, local cuisines",
                "Specialty Restaurants": "Unique cuisines, niche markets"
            }
            
            for cluster, desc in clusters.items():
                st.markdown(f"**{cluster}**: {desc}")
        
        with tab2:
            st.markdown("### Customer Sentiment Analysis")
            st.markdown("Analyze customer reviews and feedback for sentiment insights.")
            
            # Sample sentiment distribution
            sentiment_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Percentage': [65, 15, 20]
            })
            
            fig = px.pie(sentiment_data, values='Percentage', names='Sentiment', 
                         title='Review Sentiment Distribution',
                         color_discrete_sequence=['#4CAF50', '#FFC107', '#F44336'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Sample reviews
            st.markdown("#### Sample Reviews")
            reviews = [
                {"text": "The food was amazing and the service was excellent!", "sentiment": "Positive"},
                {"text": "Decent food but the wait time was too long.", "sentiment": "Neutral"},
                {"text": "Disappointing experience. Food was cold and service was poor.", "sentiment": "Negative"},
            ]
            
            for review in reviews:
                sentiment_color = "#4CAF50" if review["sentiment"] == "Positive" else "#F44336" if review["sentiment"] == "Negative" else "#FFC107"
                st.markdown(f"<div style='padding:10px; border-left:3px solid {sentiment_color}; margin-bottom:10px;'>"
                          f"<p>{review['text']}</p>"
                          f"<p style='color:{sentiment_color}; font-weight:bold;'>{review['sentiment']}</p>"
                          f"</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Market Intelligence")
            st.markdown("Comprehensive restaurant market analysis and competitive insights.")
            
            # Sample market trends
            st.markdown("#### Cuisine Popularity Trends")
            
            # Generate sample data for demonstration
            dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
            cuisines = ['Italian', 'Indian', 'Chinese', 'Mexican', 'Japanese']
            
            trend_data = pd.DataFrame()
            for cuisine in cuisines:
                trend_data[cuisine] = 50 + np.cumsum(np.random.normal(0, 5, 12))
            
            trend_data['Date'] = dates
            trend_data_melted = pd.melt(trend_data, id_vars=['Date'], value_vars=cuisines, 
                                        var_name='Cuisine', value_name='Popularity')
            
            fig = px.line(trend_data_melted, x='Date', y='Popularity', color='Cuisine',
                         title='Cuisine Popularity Trends (Last 12 Months)')
            st.plotly_chart(fig, use_container_width=True)
    
    # Song Genre Classifier Tool
    elif st.session_state.selected_tool == "song_genre":
        st.markdown('<p class="sub-header">🎵 Song Genre Classifier</p>', unsafe_allow_html=True)
        st.markdown("Identify song genres based on lyrics using machine learning")
        
        # Sample form for song genre classification
        st.markdown("### Classify Song by Lyrics")
        
        with st.form("lyrics_form"):
            lyrics = st.text_area("Enter Song Lyrics", height=200, 
                                 placeholder="Paste song lyrics here to classify the genre...")
            submitted = st.form_submit_button("Classify Genre")
        
        if submitted and lyrics:
            # Simulate classification (in a real app, this would use the trained model)
            genres = ['Country', 'Rock', 'Hip-Hop', 'Pop', 'Rhythm and Blues']
            genre_probs = np.random.dirichlet(np.ones(5), size=1)[0]
            
            # Display results
            st.markdown("### Classification Results")
            
            # Show the predicted genre
            predicted_genre = genres[np.argmax(genre_probs)]
            st.markdown(f"<div style='background-color:#f0f0f0; padding:20px; border-radius:10px;'>"
                        f"<h3 style='color:#6A1B9A; margin:0;'>Predicted Genre: {predicted_genre}</h3>"
                        f"</div>", unsafe_allow_html=True)
            
            # Show probability distribution
            st.markdown("#### Genre Probability Distribution")
            genre_df = pd.DataFrame({
                'Genre': genres,
                'Probability': genre_probs
            })
            
            fig = px.bar(genre_df, x='Genre', y='Probability', 
                         title='Genre Classification Probabilities',
                         color='Probability', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show model information
            st.markdown("#### Model Information")
            st.markdown("""
            - **Model Type**: Multinomial Naïve Bayes with Text Cleaning
            - **Accuracy**: 83%
            - **Features**: Processed lyrics with lemmatization and stopword removal
            - **Training Data**: 62,155 songs across 5 genres
            """)
    
    # YouTube Comment Analysis Tool
    elif st.session_state.selected_tool == "youtube_analysis":
        st.markdown('<p class="sub-header">📺 YouTube Comment Analysis</p>', unsafe_allow_html=True)
        st.markdown("Analyze sentiment and trends in YouTube video comments")
        
        # Sample form for YouTube video analysis
        st.markdown("### Analyze YouTube Comments")
        
        with st.form("youtube_form"):
            video_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
            submitted = st.form_submit_button("Analyze Comments")
        
        if submitted and video_url:
            # Simulate analysis (in a real app, this would fetch and analyze actual comments)
            st.markdown("### Analysis Results")
            
            # Sample metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">45%</div>
                    <div class="metric-label">Positive Comments</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">30%</div>
                    <div class="metric-label">Neutral Comments</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">25%</div>
                    <div class="metric-label">Negative Comments</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Sample sentiment over time
            st.markdown("#### Sentiment Trend Over Time")
            
            # Generate sample data for demonstration
            dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
            sentiment_trend = pd.DataFrame({
                'Date': dates,
                'Positive': 40 + np.cumsum(np.random.normal(0, 3, 30)),
                'Neutral': 30 + np.cumsum(np.random.normal(0, 2, 30)),
                'Negative': 25 + np.cumsum(np.random.normal(0, 2, 30))
            })
            
            fig = px.line(sentiment_trend, x='Date', y=['Positive', 'Neutral', 'Negative'],
                         title='Comment Sentiment Trend',
                         color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Word cloud placeholder
            st.markdown("#### Most Common Words in Comments")
            st.markdown("*Word cloud visualization would appear here in the actual implementation*")
            
            # Sample comments
            st.markdown("#### Sample Comments")
            comments = [
                {"text": "This video was so helpful! Thanks for sharing!", "sentiment": "Positive", "likes": 45},
                {"text": "Interesting content but could be more detailed.", "sentiment": "Neutral", "likes": 12},
                {"text": "Waste of time, the information is misleading.", "sentiment": "Negative", "likes": 3},
            ]
            
            for comment in comments:
                sentiment_color = "#4CAF50" if comment["sentiment"] == "Positive" else "#F44336" if comment["sentiment"] == "Negative" else "#FFC107"
                st.markdown(f"<div style='padding:10px; border-left:3px solid {sentiment_color}; margin-bottom:10px;'>"
                          f"<p>{comment['text']}</p>"
                          f"<p style='color:{sentiment_color}; font-weight:bold;'>{comment['sentiment']} • {comment['likes']} likes</p>"
                          f"</div>", unsafe_allow_html=True)
    
    # Movie Success Predictor Tool
    elif st.session_state.selected_tool == "movie_success":
        try:
            # Import and run the Movie Success Predictor module
            sys.path.append(str(Path(__file__).parent / "movie_success_predictor"))
            from app import main as movie_success_app
            
            # Run the app
            movie_success_app()
            
        except Exception as e:
            st.error(f"Error loading the Movie Success Predictor module: {str(e)}")
            st.info("Please ensure all required dependencies are installed and the module files exist.")
            
            if st.button("⬅️ Back to Dashboard"):
                go_back_to_main()
                
            # Sort factors by importance
            factors = {k: v for k, v in sorted(factors.items(), key=lambda item: item[1], reverse=True)}
            
            # Create a horizontal bar chart
            factor_df = pd.DataFrame({
                'Factor': list(factors.keys()),
                'Importance': list(factors.values())
            })
            
            fig = px.bar(factor_df, y='Factor', x='Importance', 
                         title='Feature Importance',
                         orientation='h',
                         color='Importance',
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show similar movies
            st.markdown("#### Similar Movies and Their Performance")
            
            # Generate sample similar movies
            similar_movies = [
                {"title": "Action Adventure", "budget": 110000000, "revenue": 320000000, "result": "Hit"},
                {"title": "Sci-Fi Journey", "budget": 95000000, "revenue": 210000000, "result": "Hit"},
                {"title": "Epic Quest", "budget": 150000000, "revenue": 180000000, "result": "Flop"},
                {"title": "Hero's Path", "budget": 80000000, "revenue": 250000000, "result": "Hit"},
            ]
            
            similar_df = pd.DataFrame(similar_movies)
            similar_df['ROI'] = (similar_df['revenue'] - similar_df['budget']) / similar_df['budget']
            
            fig = px.bar(similar_df, x='title', y='ROI', 
                         title='Return on Investment for Similar Movies',
                         color='result',
                         color_discrete_map={'Hit': '#4CAF50', 'Flop': '#F44336'},
                         text='result')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

else:
    # Main dashboard with metrics and tools
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("### Entertainment Industry Metrics")
        
        # Sample metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">$42.5B</div>
                <div class="metric-label">Global Box Office</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">+12.3%</div>
                <div class="metric-label">Streaming Growth</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">850M</div>
                <div class="metric-label">Music Streams</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Sample chart
        st.markdown("### Industry Trends")
        
        # Generate sample data for demonstration
        years = list(range(2018, 2024))
        sectors = {
            'Movies': [11.9, 11.4, 7.2, 8.8, 10.1, 11.5],
            'Music': [9.8, 10.7, 10.1, 12.5, 14.2, 15.9],
            'Streaming': [8.2, 10.5, 15.7, 19.2, 22.5, 25.8],
            'Gaming': [12.5, 14.2, 17.8, 19.5, 21.2, 23.5]
        }
        
        trend_data = pd.DataFrame({'Year': years})
        for sector, values in sectors.items():
            trend_data[sector] = values
        
        trend_data_melted = pd.melt(trend_data, id_vars=['Year'], value_vars=list(sectors.keys()), 
                                    var_name='Sector', value_name='Revenue (Billion $)')
        
        fig = px.line(trend_data_melted, x='Year', y='Revenue (Billion $)', color='Sector',
                     title='Entertainment Industry Revenue Trends')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Entertainment Analytics Tools")
        
        # RestaurantInsight AI
        if tool_card(
            "RestaurantInsight AI", 
            "Advanced restaurant analytics and customer intelligence platform using clustering and sentiment analysis.", 
            "🍽️", 
            "Explore Restaurant Analytics", 
            "restaurant_btn"
        ):
            st.session_state.selected_tool = "restaurant_insight"
        
        # Song Genre Classifier
        if tool_card(
            "Song Genre Classifier", 
            "Identify which genre a song belongs to based on its lyrics using machine learning.", 
            "🎵", 
            "Classify Song Genres", 
            "song_btn"
        ):
            st.session_state.selected_tool = "song_genre"
        
        # YouTube Comment Analysis
        if tool_card(
            "YouTube Comment Analysis", 
            "Analyze sentiment and trends in comments posted on YouTube videos.", 
            "📺", 
            "Analyze YouTube Comments", 
            "youtube_btn"
        ):
            st.session_state.selected_tool = "youtube_analysis"
        
        # Movie Success Predictor
        if tool_card(
            "Movie Success Predictor", 
            "Predict whether a movie will be a box office hit or flop based on various features.", 
            "🎬", 
            "Predict Movie Success", 
            "movie_btn"
        ):
            st.session_state.selected_tool = "movie_success"

# Footer
st.markdown('<div class="footer">© 2025 DataSentience-AIML | Entertainment Industry Analytics</div>', unsafe_allow_html=True)