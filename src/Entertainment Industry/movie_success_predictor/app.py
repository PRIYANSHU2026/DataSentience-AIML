"""
Movie Success Predictor

Predict whether a movie will be a box office hit or flop using machine learning.
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# Add parent directory to path for shared components
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui_components import UIComponents

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/movie_success_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# Sample data for dropdowns
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "TV Movie", "Thriller", "War", "Western"
]

# Sample popular actors and directors for autocomplete
POPULAR_ACTORS = [
    "Robert Downey Jr.", "Scarlett Johansson", "Chris Evans", "Tom Hanks",
    "Meryl Streep", "Leonardo DiCaprio", "Jennifer Lawrence", "Dwayne Johnson",
    "Emma Stone", "Ryan Reynolds", "Margot Robbie", "Tom Hardy",
    "Brad Pitt", "Angelina Jolie", "Johnny Depp"
]

POPULAR_DIRECTORS = [
    "Steven Spielberg", "Christopher Nolan", "Quentin Tarantino", "Martin Scorsese",
    "James Cameron", "Ridley Scott", "Peter Jackson", "Tim Burton",
    "Wes Anderson", "Denis Villeneuve", "Alfonso Cuarón", "Greta Gerwig"
]

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_input(movie_data):
    """Preprocess the user input to match model requirements"""
    # Create a DataFrame with all possible columns
    df = pd.DataFrame(columns=[
        'budget', 'popularity', 'runtime', 'genres_Action', 'genres_Adventure',
        'genres_Animation', 'genres_Comedy', 'genres_Crime', 'genres_Documentary',
        'genres_Drama', 'genres_Family', 'genres_Fantasy', 'genres_History',
        'genres_Horror', 'genres_Music', 'genres_Mystery', 'genres_Romance',
        'genres_Science Fiction', 'genres_TV Movie', 'genres_Thriller',
        'genres_War', 'genres_Western',
        'cast_Chris Evans', 'cast_Chris Hemsworth', 'cast_Johnny Depp',
        'cast_Leonardo DiCaprio', 'cast_Matt Damon', 'cast_Robert Downey Jr.',
        'cast_Scarlett Johansson', 'cast_Tom Cruise', 'cast_Tom Hanks',
        'director_Christopher Nolan', 'director_David Fincher',
        'director_James Cameron', 'director_Martin Scorsese',
        'director_Quentin Tarantino', 'director_Ridley Scott',
        'director_Steven Spielberg', 'director_Tim Burton',
        'director_Wes Anderson'
    ])
    
    # Initialize all columns with 0
    df.loc[0] = 0
    
    # Set numerical features
    df['budget'] = movie_data['budget']
    df['popularity'] = movie_data['popularity']
    df['runtime'] = movie_data['runtime']
    
    # Set genre (one-hot encoded)
    genre_col = f"genres_{movie_data['genres']}"
    if genre_col in df.columns:
        df[genre_col] = 1
    
    # Set cast (top actors)
    for actor in movie_data['cast'].split(','):
        actor = actor.strip()
        cast_col = f"cast_{actor}"
        if cast_col in df.columns:
            df[cast_col] = 1
    
    # Set director
    director_col = f"director_{movie_data['director']}"
    if director_col in df.columns:
        df[director_col] = 1
    
    return df

def predict_success(model, movie_data):
    """Make prediction on the input data"""
    try:
        # Preprocess input
        X = preprocess_input(movie_data)
        
        # Make prediction
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        
        return {
            'prediction': 'Hit' if prediction[0] == 1 else 'Flop',
            'confidence': max(probability[0]) * 100,
            'hit_probability': probability[0][1] * 100,
            'flop_probability': probability[0][0] * 100
        }
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def display_results(prediction_result, movie_data):
    """Display prediction results with visualizations"""
    st.success("## 🎬 Movie Success Prediction Results")
    
    # Display prediction
    if prediction_result['prediction'] == 'Hit':
        st.markdown(f"### 🎉 **Prediction: {prediction_result['prediction']}**")
        st.balloons()
    else:
        st.markdown(f"### 😕 **Prediction: {prediction_result['prediction']}**")
    
    st.write(f"**Confidence:** {prediction_result['confidence']:.1f}%")
    
    # Create gauge chart for prediction probability
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_result['hit_probability'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Hit Probability"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightcoral"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=10)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed probabilities
    st.markdown("### 📊 Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Outcome': ['Hit', 'Flop'],
        'Probability (%)': [prediction_result['hit_probability'], 
                          prediction_result['flop_probability']]
    })
    
    # Create a bar chart
    fig = px.bar(
        prob_df,
        x='Outcome',
        y='Probability (%)',
        color='Outcome',
        color_discrete_map={'Hit': 'green', 'Flop': 'red'},
        text='Probability (%)',
        height=400
    )
    
    # Update text position and format
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        textfont_size=14
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Probability (%)",
        showlegend=False,
        yaxis=dict(range=[0, 100])
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Show movie details
    with st.expander("📝 Movie Details"):
        st.write(f"**Title:** {movie_data.get('title', 'N/A')}")
        st.write(f"**Genre:** {movie_data.get('genres', 'N/A')}")
        st.write(f"**Director:** {movie_data.get('director', 'N/A')}")
        st.write(f"**Cast:** {movie_data.get('cast', 'N/A')}")
        st.write(f"**Budget:** ${movie_data.get('budget', 0):,}")
        st.write(f"**Runtime:** {movie_data.get('runtime', 0)} minutes")
        st.write(f"**Popularity Score:** {movie_data.get('popularity', 0):.1f}")

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Movie Success Predictor - Entertainment Industry",
        page_icon="🎬",
        layout="wide"
    )
    
    # Initialize session state
    if 'model' not in st.session_state:
        with st.spinner("Loading movie success prediction model..."):
            st.session_state.model = load_model()
    
    # Header
    UIComponents.header(
        "🎬 Movie Success Predictor",
        "Predict whether a movie will be a box office hit or flop using machine learning"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("🎥 About")
        st.write("""
        This tool predicts the potential success of a movie based on various factors 
        like budget, genre, cast, director, and more using machine learning.
        """)
        
        st.markdown("### 📊 How It Works")
        st.write("1. Fill in the movie details in the form")
        st.write("2. Click 'Predict Success' to analyze")
        st.write("3. View the prediction and detailed analysis")
        
        if st.button("⬅️ Back to Entertainment Industry"):
            st.switch_page("../app.py")
        
        st.markdown("---")
        st.markdown("### 🎯 Success Criteria")
        st.write("A movie is considered a 'Hit' if its revenue is expected to be at least 1.5x its budget.")
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        st.write("""
        - **Accuracy:** ~85%
        - **Features:** Budget, Genre, Cast, Director, Runtime, Popularity
        - **Algorithm:** Random Forest Classifier
        """)
    
    # Main content
    st.markdown("### 🎥 Enter Movie Details")
    
    # Create a form for movie details
    with st.form("movie_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            movie_title = st.text_input("Movie Title", "Inception")
            
            # Genre selection
            genre = st.selectbox(
                "Primary Genre",
                options=GENRES,
                index=8  # Default to Action
            )
            
            # Budget input with slider
            budget = st.slider(
                "Production Budget (in millions)",
                min_value=1,
                max_value=500,
                value=160,  # Default value
                step=1,
                help="Estimated production budget in millions of dollars"
            ) * 1000000  # Convert to actual budget
            
            # Popularity score
            popularity = st.slider(
                "Popularity Score",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=0.1,
                help="Movie popularity score (0-100)"
            )
        
        with col2:
            # Director input with autocomplete
            director = st.selectbox(
                "Director",
                options=POPULAR_DIRECTORS,
                index=1  # Default to Christopher Nolan
            )
            
            # Cast input with multi-select
            cast = st.multiselect(
                "Top Cast Members (select up to 3)",
                options=POPULAR_ACTORS,
                default=["Leonardo DiCaprio", "Tom Hardy", "Joseph Gordon-Levitt"],
                max_selections=3
            )
            
            # Runtime input
            runtime = st.slider(
                "Runtime (minutes)",
                min_value=60,
                max_value=240,
                value=148,  # Default to 2h 28min
                step=1,
                help="Movie duration in minutes"
            )
        
        # Submit button
        submitted = st.form_submit_button("🎬 Predict Success")
        
        if submitted:
            # Prepare movie data
            movie_data = {
                'title': movie_title,
                'budget': budget,
                'popularity': popularity,
                'runtime': runtime,
                'genres': genre,
                'director': director,
                'cast': ', '.join(cast)
            }
            
            # Make prediction
            if st.session_state.model is not None:
                with st.spinner("Analyzing movie details..."):
                    prediction_result = predict_success(st.session_state.model, movie_data)
                    
                    if prediction_result:
                        # Store results in session state
                        st.session_state.prediction_result = prediction_result
                        st.session_state.movie_data = movie_data
                        
                        # Scroll to results
                        st.experimental_rerun()
            else:
                st.error("Failed to load the prediction model. Please try again later.")
    
    # Display results if available
    if 'prediction_result' in st.session_state and 'movie_data' in st.session_state:
        st.markdown("---")
        display_results(
            st.session_state.prediction_result,
            st.session_state.movie_data
        )
        
        # Add export functionality
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        # Create a DataFrame with the results
        results_df = pd.DataFrame([{
            'Movie Title': st.session_state.movie_data['title'],
            'Prediction': st.session_state.prediction_result['prediction'],
            'Hit Probability (%)': st.session_state.prediction_result['hit_probability'],
            'Flop Probability (%)': st.session_state.prediction_result['flop_probability'],
            'Confidence (%)': st.session_state.prediction_result['confidence'],
            'Genre': st.session_state.movie_data['genres'],
            'Director': st.session_state.movie_data['director'],
            'Budget ($)': st.session_state.movie_data['budget'],
            'Runtime (min)': st.session_state.movie_data['runtime'],
            'Popularity': st.session_state.movie_data['popularity']
        }])
        
        # Convert DataFrame to CSV
        csv = results_df.to_csv(index=False)
        
        # Download button
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv,
            file_name=f"movie_success_prediction_{st.session_state.movie_data['title'].replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        # Show tips based on prediction
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        if st.session_state.prediction_result['prediction'] == 'Hit':
            st.success("""
            🎯 **Your movie has high hit potential!**
            
            Consider these next steps:
            - **Marketing Strategy**: Focus on highlighting the star cast and director
            - **Release Timing**: Choose a release date with less competition
            - **Target Audience**: Tailor marketing to fans of this genre
            - **Merchandising**: Develop merchandise to maximize revenue
            """)
        else:
            st.warning("""
            ⚠️ **Your movie might face challenges at the box office.**
            
            Consider these improvements:
            - **Budget Review**: Can production costs be optimized?
            - **Cast & Crew**: Would adding a well-known actor or director help?
            - **Genre Appeal**: Is there a way to broaden the movie's appeal?
            - **Marketing**: Consider a strong marketing campaign to build awareness
            - **Release Strategy**: A festival run might help build buzz
            """)

if __name__ == "__main__":
    main()