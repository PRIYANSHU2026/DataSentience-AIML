"""
Advanced Duplicate Question Detection Platform
A comprehensive web application for detecting duplicate questions using multiple ML techniques
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import time
import io
import base64

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AdvancedDuplicateDetector:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_features(self, q1, q2):
        """Extract comprehensive features for duplicate detection"""
        features = {}
        
        # Basic length features
        features['q1_length'] = len(q1)
        features['q2_length'] = len(q2)
        features['length_diff'] = abs(len(q1) - len(q2))
        features['length_ratio'] = min(len(q1), len(q2)) / max(len(q1), len(q2)) if max(len(q1), len(q2)) > 0 else 0
        
        # Word-level features
        q1_words = set(q1.split())
        q2_words = set(q2.split())
        
        features['common_words'] = len(q1_words.intersection(q2_words))
        features['total_words'] = len(q1_words.union(q2_words))
        features['word_share'] = features['common_words'] / features['total_words'] if features['total_words'] > 0 else 0
        
        # Fuzzy matching features
        features['fuzz_ratio'] = fuzz.ratio(q1, q2) / 100.0
        features['fuzz_partial_ratio'] = fuzz.partial_ratio(q1, q2) / 100.0
        features['fuzz_token_sort_ratio'] = fuzz.token_sort_ratio(q1, q2) / 100.0
        features['fuzz_token_set_ratio'] = fuzz.token_set_ratio(q1, q2) / 100.0
        
        return features
    
    def calculate_similarity(self, q1, q2):
        """Calculate similarity score using multiple methods"""
        # Preprocess questions
        q1_processed = self.preprocess_text(q1)
        q2_processed = self.preprocess_text(q2)
        
        # Extract features
        features = self.extract_features(q1, q2)
        
        # TF-IDF similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([q1_processed, q2_processed])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_similarity = 0.0
        
        # Weighted scoring
        weights = {
            'tfidf': 0.4,
            'fuzzy': 0.3,
            'word_share': 0.2,
            'length': 0.1
        }
        
        fuzzy_score = (features['fuzz_ratio'] + features['fuzz_token_sort_ratio'] + features['fuzz_token_set_ratio']) / 3
        
        final_score = (
            weights['tfidf'] * tfidf_similarity +
            weights['fuzzy'] * fuzzy_score +
            weights['word_share'] * features['word_share'] +
            weights['length'] * features['length_ratio']
        )
        
        return final_score, features, tfidf_similarity, fuzzy_score

def create_feature_comparison_chart(features):
    """Create a radar chart showing feature comparison"""
    categories = ['Length Ratio', 'Word Share', 'Fuzz Ratio', 'Token Sort', 'Token Set', 'TF-IDF']
    values = [
        features['length_ratio'],
        features['word_share'],
        features['fuzz_ratio'],
        features['fuzz_token_sort_ratio'],
        features['fuzz_token_set_ratio'],
        features.get('tfidf_similarity', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Similarity Metrics',
        line_color='rgb(106, 138, 255)',
        fillcolor='rgba(106, 138, 255, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Similarity Analysis Breakdown",
        height=400
    )
    
    return fig

def create_confidence_gauge(score):
    """Create a gauge chart for confidence score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Similarity Confidence (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="Advanced Duplicate Question Detector",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Advanced Duplicate Question Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">State-of-the-art AI-powered duplicate question detection using multiple machine learning techniques</p>', unsafe_allow_html=True)
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = AdvancedDuplicateDetector()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
        show_details = st.checkbox("Show Detailed Analysis", value=True)
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("- TF-IDF Vectorization")
        st.markdown("- Fuzzy String Matching")
        st.markdown("- Word-level Analysis")
        st.markdown("- Length-based Features")
        st.markdown("- Real-time Processing")
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Question 1")
        q1 = st.text_area(
            "",
            placeholder="Enter the first question...",
            height=100,
            key="q1"
        )
    
    with col2:
        st.subheader("📝 Question 2")
        q2 = st.text_area(
            "",
            placeholder="Enter the second question...",
            height=100,
            key="q2"
        )
    
    # Analysis button
    if st.button("🔍 Analyze Similarity", type="primary"):
        if q1.strip() and q2.strip():
            with st.spinner("Analyzing questions..."):
                # Simulate processing time for better UX
                time.sleep(1)
                
                # Calculate similarity
                score, features, tfidf_sim, fuzzy_score = st.session_state.detector.calculate_similarity(q1, q2)
                
                # Results section
                st.markdown("---")
                st.subheader("📈 Analysis Results")
                
                # Main result
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    is_duplicate = score >= threshold
                    status = "✅ DUPLICATE" if is_duplicate else "❌ NOT DUPLICATE"
                    color = "green" if is_duplicate else "red"
                    st.markdown(f"<h3 style='text-align: center; color: {color};'>{status}</h3>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Similarity Score", f"{score:.2%}", f"{(score - threshold):.2%}")
                
                with col3:
                    confidence = "High" if abs(score - threshold) > 0.2 else "Medium" if abs(score - threshold) > 0.1 else "Low"
                    st.metric("Confidence", confidence, f"Threshold: {threshold:.1%}")
                
                if show_details:
                    st.markdown("---")
                    
                    # Detailed analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Feature comparison radar chart
                        features['tfidf_similarity'] = tfidf_sim
                        radar_chart = create_feature_comparison_chart(features)
                        st.plotly_chart(radar_chart, use_container_width=True)
                    
                    with col2:
                        # Confidence gauge
                        gauge_chart = create_confidence_gauge(score)
                        st.plotly_chart(gauge_chart, use_container_width=True)
                    
                    # Feature breakdown table
                    st.subheader("🔍 Detailed Feature Analysis")
                    
                    feature_df = pd.DataFrame([
                        ["Text Length (Q1)", features['q1_length']],
                        ["Text Length (Q2)", features['q2_length']],
                        ["Length Difference", features['length_diff']],
                        ["Length Ratio", f"{features['length_ratio']:.3f}"],
                        ["Common Words", features['common_words']],
                        ["Word Share Ratio", f"{features['word_share']:.3f}"],
                        ["Fuzzy Ratio", f"{features['fuzz_ratio']:.3f}"],
                        ["Token Sort Ratio", f"{features['fuzz_token_sort_ratio']:.3f}"],
                        ["Token Set Ratio", f"{features['fuzz_token_set_ratio']:.3f}"],
                        ["TF-IDF Similarity", f"{tfidf_sim:.3f}"],
                    ], columns=["Feature", "Value"])
                    
                    st.dataframe(feature_df, use_container_width=True)
        else:
            st.warning("⚠️ Please enter both questions to analyze.")
    
    # Batch processing section
    st.markdown("---")
    st.subheader("📊 Batch Processing")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with question pairs",
        type=['csv'],
        help="CSV should have columns: 'question1', 'question2'"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Process Batch"):
                if 'question1' in df.columns and 'question2' in df.columns:
                    with st.spinner("Processing batch..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            score, _, _, _ = st.session_state.detector.calculate_similarity(
                                str(row['question1']), str(row['question2'])
                            )
                            results.append({
                                'Question 1': row['question1'],
                                'Question 2': row['question2'],
                                'Similarity Score': score,
                                'Is Duplicate': score >= threshold
                            })
                            progress_bar.progress((idx + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        st.success(f"Processed {len(results_df)} question pairs!")
                        
                        # Show results
                        st.dataframe(results_df)
                        
                        # Download results
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_data,
                            file_name="duplicate_detection_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("CSV must contain 'question1' and 'question2' columns")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🚀 Built with Streamlit | 🤖 Powered by Machine Learning | 💡 Enhanced with Advanced NLP</p>
        <p>Features: TF-IDF Vectorization • Fuzzy Matching • Real-time Analysis • Batch Processing</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
