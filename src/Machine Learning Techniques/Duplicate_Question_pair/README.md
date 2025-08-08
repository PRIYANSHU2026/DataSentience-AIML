# 🔍 Duplicate Question Pair Detection

An advanced NLP project showcasing the evolution from basic ML to production-ready web applications for semantic question similarity detection.

## 📊 Dataset
**Source**: [Kaggle Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) (400K+ question pairs)

## 🚀 Project Components

### 📓 Research Notebooks
Progressive ML development through multiple approaches:

1. **`only-bow.ipynb`**: Basic Bag of Words + Random Forest (75% accuracy)
2. **`bow-with-basic-features.ipynb`**: Enhanced with 7 additional features (80% accuracy)
3. **`bow-with-preprocessing-and-advanced-features.ipynb`**: Advanced NLP preprocessing (90% accuracy)
4. **`initial_EDA.ipynb`**: Exploratory data analysis

### 🌐 Production Web Application
**[Advanced Streamlit App](./streamlit-app/)**: Professional duplicate detection platform

#### Key Features:
- Multi-algorithm similarity detection (TF-IDF + Fuzzy matching)
- Interactive dashboard with real-time analysis
- Customizable similarity thresholds  
- Batch processing capabilities
- Advanced visualizations (radar charts, confidence gauges)
- Mobile-responsive design

#### Quick Start:
```bash
cd streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

**📖 [Full Documentation & Features →](./streamlit-app/README.md)**

## 📈 Performance Evolution

| Stage | Approach | Accuracy | Implementation |
|-------|----------|----------|----------------|
| 1 | Basic BoW | 75% | Random Forest |
| 2 | Enhanced Features | 80% | + 7 custom features |
| 3 | Advanced NLP | 90% | Full preprocessing |
| 4 | **Production App** | **Real-time** | **Multi-algorithm fusion** |

## 🎯 Learning Path

1. **Start with notebooks** to understand ML progression
2. **Test the web app** with your own question pairs
3. **Deploy to production** using included configs

---

**🚀 Complete ML Pipeline**: From research notebooks to production-ready web application

