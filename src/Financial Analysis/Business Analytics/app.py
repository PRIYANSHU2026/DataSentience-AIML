import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Configure page
st.set_page_config(
    page_title="Business Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #43A047;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #43A047;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .card-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #37474F;
    }
    .card-description {
        color: #546E7A;
        margin-bottom: 1rem;
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

# Header
st.markdown('<div class="main-header">📊 Business Analytics Dashboard</div>', unsafe_allow_html=True)

# Function to create a card with hover effect
def create_card(title, description, key, icon):
    st.markdown(f'''
    <div class="card" id="{key}-card">
        <div class="card-title">{icon} {title}</div>
        <div class="card-description">{description}</div>
    </div>
    ''', unsafe_allow_html=True)
    return st.button(f"Explore {title}", key=key)

# Sidebar navigation
st.sidebar.title("Business Analytics Tools")
selected_tool = st.sidebar.radio(
    "Select a tool:",
    ["Dashboard", "Startup Success Predictor", "Startup Profit Prediction", "Employee Attrition", "Statement Analysis", "Resume Filtering", "Interview Success", "Airbnb Analytics"]
)

# Main content
if selected_tool == "Dashboard":
    st.subheader("Business Analytics Overview")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Startup Success Rate", "68%", "+4%")
    with col2:
        st.metric("Average Profit Margin", "23.5%", "-1.2%")
    with col3:
        st.metric("Employee Retention", "82%", "+2.5%")
    
    # Sample charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Startup funding by sector
        st.subheader("Startup Funding by Sector")
        sectors = ['Technology', 'Healthcare', 'Finance', 'Education', 'Retail', 'Manufacturing']
        funding = [250, 180, 120, 80, 60, 40]
        
        fig = px.bar(
            x=sectors,
            y=funding,
            labels={'x': 'Sector', 'y': 'Funding ($ millions)'},
            color=funding,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Employee attrition factors
        st.subheader("Employee Attrition Factors")
        factors = ['Salary', 'Work-Life Balance', 'Career Growth', 'Management', 'Company Culture', 'Benefits']
        impact = [35, 25, 20, 10, 7, 3]
        
        fig = px.pie(
            values=impact,
            names=factors,
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Business tools showcase
    st.subheader("Business Analytics Tools")
    
    tool_col1, tool_col2 = st.columns(2)
    
    with tool_col1:
        create_card(
            "Startup Success Predictor", 
            "ML-powered platform to evaluate startup potential and success factors.", 
            "startup_success_card", 
            "🚀"
        )
        
        create_card(
            "Employee Attrition Prediction", 
            "Predict and analyze factors affecting employee retention.", 
            "employee_attrition_card", 
            "👥"
        )
        
        create_card(
            "Resume Filtering Assistant", 
            "AI-powered tool to filter and rank job applicants based on resume analysis.", 
            "resume_filtering_card", 
            "📄"
        )
    
    with tool_col2:
        create_card(
            "Startup Profit Prediction", 
            "Analyze and predict startup profit based on key business metrics.", 
            "profit_prediction_card", 
            "💰"
        )
        
        create_card(
            "Statement Analysis", 
            "Analyze financial statements and news sentiment for market impact.", 
            "statement_analysis_card", 
            "📑"
        )
        
        create_card(
            "Interview Success Predictor", 
            "Predict interview success probability based on candidate profiles.", 
            "interview_success_card", 
            "🤝"
        )

elif selected_tool == "Startup Success Predictor":
    st.subheader("🚀 Startup Success Predictor")
    
    # Simulated startup success prediction tool
    st.write("Predict the success probability of a startup based on key factors.")
    
    with st.form("startup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Education", "Retail", "Manufacturing"])
            funding = st.number_input("Initial Funding ($ thousands)", 10, 10000, 500)
            team_size = st.slider("Team Size", 1, 100, 10)
            founder_experience = st.slider("Founder Experience (years)", 0, 20, 5)
        
        with col2:
            has_mvp = st.selectbox("Has MVP (Minimum Viable Product)", ["Yes", "No"])
            location = st.selectbox("Location", ["Silicon Valley", "New York", "London", "Berlin", "Singapore", "Other"])
            business_model = st.selectbox("Business Model", ["B2B", "B2C", "B2B2C", "Marketplace", "SaaS"])
            market_size = st.selectbox("Target Market Size", ["Small", "Medium", "Large", "Very Large"])
        
        submitted = st.form_submit_button("Predict Success Probability")
    
    if submitted:
        # Simulate prediction based on inputs
        base_score = 50
        
        # Industry factor
        industry_scores = {"Technology": 15, "Healthcare": 12, "Finance": 10, "Education": 8, "Retail": 6, "Manufacturing": 5}
        industry_factor = industry_scores.get(industry, 5)
        
        # Funding factor (0-20 points)
        funding_factor = min(20, funding / 500)
        
        # Team size factor (0-10 points)
        team_factor = min(10, team_size / 10)
        
        # Founder experience (0-15 points)
        exp_factor = min(15, founder_experience * 0.75)
        
        # MVP factor
        mvp_factor = 10 if has_mvp == "Yes" else 0
        
        # Location factor
        location_scores = {"Silicon Valley": 10, "New York": 8, "London": 7, "Berlin": 6, "Singapore": 6, "Other": 3}
        location_factor = location_scores.get(location, 3)
        
        # Business model factor
        model_scores = {"SaaS": 10, "Marketplace": 8, "B2B": 7, "B2B2C": 6, "B2C": 5}
        model_factor = model_scores.get(business_model, 5)
        
        # Market size factor
        market_scores = {"Very Large": 10, "Large": 8, "Medium": 5, "Small": 2}
        market_factor = market_scores.get(market_size, 5)
        
        # Calculate final score (cap at 95%)
        success_score = min(95, base_score + industry_factor + funding_factor + team_factor + 
                          exp_factor + mvp_factor + location_factor + model_factor + market_factor)
        
        # Display result with gauge chart
        st.subheader("Startup Success Prediction")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = success_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Success Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 30], 'color': "#EF5350"},
                    {'range': [30, 60], 'color': "#FFCA28"},
                    {'range': [60, 100], 'color': "#66BB6A"}
                ]
            }
        ))
        st.plotly_chart(fig)
        
        # Key factors analysis
        st.subheader("Key Success Factors")
        
        factors_df = pd.DataFrame({
            'Factor': ['Industry', 'Funding', 'Team Size', 'Founder Experience', 'MVP', 'Location', 'Business Model', 'Market Size'],
            'Impact': [industry_factor, funding_factor, team_factor, exp_factor, mvp_factor, location_factor, model_factor, market_factor],
            'Max Impact': [15, 20, 10, 15, 10, 10, 10, 10]
        })
        
        fig = px.bar(
            factors_df,
            y='Factor',
            x='Impact',
            orientation='h',
            labels={'Impact': 'Impact on Success', 'Factor': ''},
            color='Impact',
            color_continuous_scale='Viridis',
            text='Impact'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on score
        st.subheader("Recommendations")
        
        if success_score >= 80:
            st.success("Your startup has excellent potential for success! Focus on scaling and market expansion.")
        elif success_score >= 60:
            st.info("Your startup shows good potential. Consider strengthening your team and refining your MVP.")
        elif success_score >= 40:
            st.warning("Your startup has moderate potential. Work on securing more funding and improving your business model.")
        else:
            st.error("Your startup faces significant challenges. Reconsider your market approach and business fundamentals.")

elif selected_tool == "Startup Profit Prediction":
    st.subheader("💰 Startup Profit Prediction")
    
    # Simulated startup profit prediction tool based on the README
    st.write("Predict the profit of a startup based on key business metrics.")
    
    with st.form("profit_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            rd_spend = st.number_input("R&D Spend ($)", 0, 1000000, 100000)
            admin_spend = st.number_input("Administration Spend ($)", 0, 500000, 80000)
        
        with col2:
            marketing_spend = st.number_input("Marketing Spend ($)", 0, 500000, 200000)
            state = st.selectbox("State", ["New York", "California", "Florida"])
        
        submitted = st.form_submit_button("Predict Profit")
    
    if submitted:
        # Simulate prediction based on a simple linear model
        # This is a simplified version of what would be a trained model in production
        base_profit = 50000
        rd_factor = rd_spend * 0.15
        admin_factor = admin_spend * 0.05
        marketing_factor = marketing_spend * 0.1
        
        state_factors = {"California": 20000, "New York": 15000, "Florida": 10000}
        state_factor = state_factors.get(state, 10000)
        
        predicted_profit = base_profit + rd_factor + marketing_factor - admin_factor + state_factor
        
        # Display result
        st.subheader("Predicted Annual Profit")
        st.metric("Predicted Profit", f"${predicted_profit:,.2f}")
        
        # Visualize contribution of each factor
        st.subheader("Profit Contribution Factors")
        
        factors = ['Base Profit', 'R&D Impact', 'Marketing Impact', 'Admin Impact', 'State Impact']
        values = [base_profit, rd_factor, marketing_factor, -admin_factor, state_factor]
        colors = ['#66BB6A', '#42A5F5', '#FFCA28', '#EF5350', '#AB47BC']
        
        fig = go.Figure(go.Waterfall(
            name = "Profit Breakdown",
            orientation = "v",
            measure = ["absolute", "relative", "relative", "relative", "relative", "total"],
            x = factors + ['Total Profit'],
            y = values + [predicted_profit],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            increasing = {"marker":{"color":"#66BB6A"}},
            decreasing = {"marker":{"color":"#EF5350"}},
            text = [f"${v:,.2f}" for v in values] + [f"${predicted_profit:,.2f}"],
            textposition = "outside"
        ))
        
        fig.update_layout(title = "Profit Contribution Waterfall", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Business Recommendations")
        
        recommendations = []
        
        if rd_spend < 150000:
            recommendations.append("Consider increasing R&D investment to drive innovation and long-term growth.")
        
        if marketing_spend < 100000:
            recommendations.append("Your marketing budget may be insufficient for optimal market reach.")
        elif marketing_spend > 300000:
            recommendations.append("Your marketing spend is high - ensure you're measuring ROI effectively.")
        
        if admin_spend > 120000:
            recommendations.append("Look for opportunities to optimize administrative costs.")
        
        if not recommendations:
            recommendations.append("Your current spending allocation appears well-balanced.")
        
        for rec in recommendations:
            st.info(rec)

# Add more tools as needed for other business analytics components

# Footer
st.markdown('<div class="footer">© 2025 DataSentience-AIML | Business Analytics Module</div>', unsafe_allow_html=True)