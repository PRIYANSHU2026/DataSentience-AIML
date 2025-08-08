# 🔍 Advanced Duplicate Question Detector

A state-of-the-art web application for detecting duplicate questions using multiple machine learning techniques and natural language processing.

## 🌟 Features

### Core Functionality
- **Multi-Algorithm Detection**: Combines TF-IDF vectorization, fuzzy string matching, and word-level analysis
- **Real-time Processing**: Instant similarity analysis with confidence scoring
- **Interactive Dashboard**: Beautiful, responsive UI with detailed visualizations
- **Batch Processing**: Upload CSV files to process multiple question pairs simultaneously
- **Confidence Scoring**: Advanced scoring system with customizable thresholds

### Advanced Analytics
- **Radar Chart Analysis**: Visual breakdown of similarity metrics
- **Confidence Gauge**: Real-time confidence indicator
- **Feature Breakdown**: Detailed analysis of all similarity features
- **Downloadable Results**: Export batch processing results as CSV

### Technical Features
- **Cross-platform Compatibility**: Works on Windows, macOS, and Linux
- **Mobile Responsive**: Optimized for mobile and tablet devices
- **Performance Optimized**: Efficient processing for large datasets
- **Error Handling**: Robust error handling and user feedback

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SiyaDadpe/DataSentience-AIML.git
   cd DataSentience-AIML/src/Machine\ Learning\ Techniques/Duplicate_Question_pair/streamlit-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## 📊 How It Works

### Similarity Detection Pipeline

1. **Text Preprocessing**
   - Lowercase conversion
   - Special character removal
   - Tokenization
   - Stopword removal
   - Stemming

2. **Feature Extraction**
   - **Length Features**: Character count, word count, length ratios
   - **Word-level Features**: Common words, word share ratios
   - **Fuzzy Features**: Ratio, partial ratio, token sort, token set
   - **TF-IDF Features**: Vectorization and cosine similarity

3. **Scoring Algorithm**
   - Weighted combination of all features
   - Customizable threshold settings
   - Confidence calculation

### Supported Input Formats

- **Single Pair Analysis**: Direct text input for two questions
- **Batch Processing**: CSV file with columns `question1` and `question2`

## 🎯 Use Cases

- **Educational Platforms**: Identify duplicate questions in Q&A systems
- **Content Management**: Detect repetitive content in forums
- **Customer Support**: Consolidate similar support tickets
- **Research**: Analyze question similarity in surveys and studies
- **Data Cleaning**: Remove duplicate entries from datasets

## 📈 Performance Metrics

The system uses multiple similarity metrics:

- **TF-IDF Cosine Similarity**: Semantic similarity based on term frequency
- **Fuzzy String Matching**: Character-level similarity analysis
- **Word Share Ratio**: Proportion of common words
- **Length Similarity**: Relative length comparison

## 🛠️ Technical Stack

- **Frontend**: Streamlit with custom CSS
- **Backend**: Python with scikit-learn
- **NLP**: NLTK for text processing
- **Visualization**: Plotly for interactive charts
- **String Matching**: FuzzyWuzzy for approximate matching

## 📱 User Interface

### Main Features
- **Dual Input Interface**: Side-by-side question entry
- **Real-time Analysis**: Instant similarity calculation
- **Visual Results**: Color-coded duplicate/non-duplicate indicators
- **Detailed Breakdown**: Comprehensive feature analysis

### Visualization Components
- **Similarity Radar Chart**: Multi-dimensional similarity analysis
- **Confidence Gauge**: Visual confidence indicator
- **Feature Table**: Detailed numeric breakdown
- **Batch Results**: Sortable and filterable results table

## 🔧 Configuration

### Adjustable Parameters
- **Similarity Threshold**: Customize the duplicate detection threshold (0.0 - 1.0)
- **Feature Weights**: Modify the importance of different similarity metrics
- **TF-IDF Settings**: Adjust vectorization parameters
- **Preprocessing Options**: Enable/disable stemming, stopword removal

## 📊 Example Usage

### Single Question Analysis
```python
Question 1: "How do I learn Python programming?"
Question 2: "What's the best way to learn Python?"

Result: ✅ DUPLICATE (Similarity: 87.3%)
```

### Batch Processing
Upload a CSV file with question pairs and get comprehensive results including:
- Similarity scores for all pairs
- Duplicate/non-duplicate classifications
- Downloadable results in CSV format

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **scikit-learn**: For machine learning algorithms
- **NLTK**: For natural language processing
- **Streamlit**: For the web application framework
- **Plotly**: For interactive visualizations
- **FuzzyWuzzy**: For string matching algorithms

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the `/docs` folder
- Review the FAQ section below

## ❓ FAQ

**Q: What makes this different from basic string matching?**
A: Our system combines multiple ML techniques including TF-IDF vectorization, fuzzy matching, and word-level analysis for more accurate results.

**Q: Can I process large datasets?**
A: Yes! The batch processing feature can handle thousands of question pairs efficiently.

**Q: Is the similarity threshold customizable?**
A: Absolutely! You can adjust the threshold in the sidebar to fine-tune sensitivity.

**Q: Does it work with languages other than English?**
A: Currently optimized for English, but can be extended for other languages by modifying the preprocessing pipeline.

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] API endpoint for programmatic access
- [ ] Advanced ML models (BERT, transformer-based)
- [ ] Real-time collaboration features
- [ ] Integration with popular Q&A platforms
- [ ] Performance analytics dashboard

---

**Built with ❤️ for the open-source community**
