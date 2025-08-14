import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Credit Card Eligibility Checker",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import eligibility logic functions
sys.path.append(os.path.join(os.path.dirname(__file__), "Scripts"))
from eligibility_logic import check_eligibility, calculate_credit_score, get_card_recommendations, get_financial_tips

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FB8C00;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #fff8e1 0%, #ffe0b2 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #EF6C00;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #FB8C00;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .credit-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 18px;
        margin: 1rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .credit-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.1);
        transform: rotate(45deg);
    }
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #e8f5e9 100%);
        color: #155724;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 3px 10px rgba(40, 167, 69, 0.1);
    }
    .warning-message {
        background: linear-gradient(135deg, #f8d7da 0%, #fdeaea 100%);
        color: #721c24;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #dc3545;
        box-shadow: 0 3px 10px rgba(220, 53, 69, 0.1);
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
st.markdown('<div class="main-header">💳 Credit Card Eligibility Checker</div>', unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.header("📊 Quick Facts")
    st.info("""
    **Minimum Requirements:**
    - Age: 18+ years
    - Income: ₹20,000/month
    - Employment: Salaried/Self-employed
    """)
    
    st.header("💡 Tips")
    st.success("""
    **Improve Your Eligibility:**
    - Maintain stable income
    - Build credit history
    - Keep debt-to-income ratio low
    """)
    
    st.header("🏦 Card Types")
    st.write("""
    - **Basic**: ₹20K+ income
    - **Silver**: ₹50K+ income  
    - **Gold**: ₹100K+ income
    - **Platinum**: ₹200K+ income
    """)

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Personal Information")
    
    # Input Form
    with st.form("eligibility_form"):
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            age = st.number_input(
                "🎂 Age",
                min_value=18,
                max_value=80,
                value=30,
                step=1,
                help="Enter your current age (18-80 years)"
            )
            
            employment_status = st.selectbox(
                "💼 Employment Status",
                ["Salaried", "Self-employed", "Student", "Unemployed", "Retired"],
                help="Select your current employment status"
            )
            
            income = st.number_input(
                "💰 Monthly Income (₹)",
                min_value=0,
                max_value=10000000,
                value=50000,
                step=5000,
                help="Enter your average monthly income in Rupees"
            )
        
        with input_col2:
            experience = st.slider(
                "⏳ Work Experience (Years)",
                0, 40, 5,
                help="Years of relevant work experience"
            )
            
            existing_cards = st.number_input(
                "💳 Existing Credit Cards",
                0, 10, 1,
                help="Number of credit cards you currently hold"
            )
            
            city_tier = st.selectbox(
                "🏙️ City Tier",
                ["Tier 1 (Metro)", "Tier 2", "Tier 3"],
                help="Classification of your city of residence"
            )
        
        submitted = st.form_submit_button("Check Eligibility")
    st.markdown('</div>', unsafe_allow_html=True)

    # Results Section (displayed after form submission)
    if submitted:
        # Call the eligibility checking function
        eligible, message, tier = check_eligibility(
            age, income, employment_status, experience, existing_cards, city_tier
        )
        
        if eligible:
            st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)
            
            # Credit Score Calculation
            credit_score = calculate_credit_score(
                age, income, employment_status, experience, existing_cards, city_tier
            )
            
            # Credit Score Gauge
            st.subheader("📊 Estimated Credit Score")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = credit_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Credit Score", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "royalblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [300, 580], 'color': "#EF5350"},
                        {'range': [580, 670], 'color': "#FFCA28"},
                        {'range': [670, 740], 'color': "#66BB6A"},
                        {'range': [740, 850], 'color': "#43A047"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 600
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Credit score interpretation
            if credit_score >= 740:
                st.success("Excellent credit score! You qualify for premium credit cards with the best terms.")
            elif credit_score >= 670:
                st.success("Good credit score. You qualify for most credit cards with favorable terms.")
            elif credit_score >= 580:
                st.warning("Fair credit score. You may qualify for standard credit cards.")
            else:
                st.error("Poor credit score. You may face challenges in credit card approval.")
            
            # Card Recommendations
            st.subheader("💳 Recommended Credit Cards")
            
            # Display card recommendations based on tier
            if tier == "Platinum":
                st.markdown("""
                <div class="credit-card" style="background: linear-gradient(135deg, #000000 0%, #434343 100%);">  
                    <h3>Platinum Rewards</h3>
                    <p>Premium travel benefits, concierge service, and exclusive rewards</p>
                    <ul>
                        <li>Annual Fee: ₹10,000</li>
                        <li>Rewards: 4x points on travel & dining</li>
                        <li>Welcome Bonus: 50,000 points</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif tier == "Gold":
                st.markdown("""
                <div class="credit-card" style="background: linear-gradient(135deg, #DAA520 0%, #B8860B 100%);">  
                    <h3>Gold Rewards</h3>
                    <p>Enhanced rewards and travel benefits</p>
                    <ul>
                        <li>Annual Fee: ₹5,000</li>
                        <li>Rewards: 3x points on travel & dining</li>
                        <li>Welcome Bonus: 25,000 points</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif tier == "Silver":
                st.markdown("""
                <div class="credit-card" style="background: linear-gradient(135deg, #C0C0C0 0%, #A9A9A9 100%);">  
                    <h3>Silver Rewards</h3>
                    <p>Balanced rewards and benefits</p>
                    <ul>
                        <li>Annual Fee: ₹2,500</li>
                        <li>Rewards: 2x points on select categories</li>
                        <li>Welcome Bonus: 10,000 points</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:  # Basic
                st.markdown("""
                <div class="credit-card" style="background: linear-gradient(135deg, #4682B4 0%, #5F9EA0 100%);">  
                    <h3>Basic Rewards</h3>
                    <p>Entry-level card with essential benefits</p>
                    <ul>
                        <li>Annual Fee: ₹1,000 (waived first year)</li>
                        <li>Rewards: 1x points on all purchases</li>
                        <li>Welcome Bonus: 5,000 points</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Financial Tips
            st.subheader("💡 Financial Improvement Tips")
            tips = get_financial_tips(age, income, employment_status, credit_score, tier)
            
            for tip in tips:
                st.info(tip)
        else:
            st.markdown(f'<div class="warning-message">{message}</div>', unsafe_allow_html=True)
            st.subheader("💡 Eligibility Improvement Tips")
            
            if age < 18:
                st.info("You must be at least 18 years old to apply for a credit card.")
            elif age > 65:
                st.info("Consider applying for a senior citizen special card with a co-applicant.")
            
            if income < 15000:
                st.info("Work on increasing your income to at least ₹15,000 per month.")
                st.info("Consider applying for a secured credit card as an alternative.")
            
            if employment_status.lower() not in ['salaried', 'self-employed']:
                st.info("Banks typically require applicants to be either salaried or self-employed.")
                st.info("If you're a student, look for student-specific credit card options.")

with col2:
    # Sidebar content
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Credit Score Factors")
    
    factors = [
        "Payment History (35%)",
        "Credit Utilization (30%)",
        "Credit History Length (15%)",
        "Credit Mix (10%)",
        "New Credit (10%)"
    ]
    
    importance = [35, 30, 15, 10, 10]
    
    fig = px.pie(
        values=importance,
        names=factors,
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Eligibility criteria
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✅ Eligibility Criteria")
    
    st.markdown("""
    - **Age**: 18-65 years
    - **Income**: Minimum ₹15,000/month
    - **Employment**: Salaried or Self-employed
    - **Credit Score**: Preferably 650+
    - **Documentation**: ID proof, Address proof, Income proof
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Card comparison
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔄 Card Tier Comparison")
    
    comparison_data = pd.DataFrame({
        'Feature': ['Annual Fee', 'Rewards Rate', 'Welcome Bonus', 'Foreign Transaction Fee', 'Airport Lounge Access'],
        'Basic': ['₹1,000', '1x points', '5,000 points', '3%', 'No'],
        'Silver': ['₹2,500', '2x points', '10,000 points', '2.5%', 'Limited'],
        'Gold': ['₹5,000', '3x points', '25,000 points', '2%', 'Yes'],
        'Platinum': ['₹10,000', '4x points', '50,000 points', 'None', 'Unlimited']
    })
    
    st.dataframe(comparison_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 DataSentience-AIML | Credit Card Eligibility Module</div>', unsafe_allow_html=True)