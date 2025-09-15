import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os

# --- Custom CSS for Professional Look ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: #e8e8e8;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .section-header {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .section-header h3 {
        color: #2c3e50;
        margin: 0;
        font-weight: 600;
    }
    .best-model-banner {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .churn-yes {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        box-shadow: 0 4px 8px rgba(255,107,107,0.3);
    }
    .churn-no {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        box-shadow: 0 4px 8px rgba(81,207,102,0.3);
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    .input-summary {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .input-summary h4 {
        color: #2c3e50 !important;
        margin-bottom: 1rem;
    }
    .input-summary ul {
        list-style-type: none;
        padding: 0;
    }
    .input-summary li {
        color: #495057;
        margin: 0.5rem 0;
        padding: 0.3rem 0;
        border-bottom: 1px solid #dee2e6;
    }
    .input-summary strong {
        color: #2c3e50;
    }
    .developer-section {
        background: #f1f3f4;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 3rem;
        border: 2px dashed #6c757d;
    }
    /*
    * This is the fix for the text color on white background
    */
    .metric-container h4, .metric-container p {
        color: #2c3e50 !important;
    }
    .metric-container h2 {
        color: #667eea !important;
    }
    .metric-container h2.risk-high {
        color: #dc3545 !important;
    }
    .metric-container h2.risk-medium {
        color: #ffc107 !important;
    }
    .metric-container h2.risk-low {
        color: #28a745 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_churn_model.pkl")
COMPARISON_REPORT_PATH = "reports/model_comparison.csv"

# --- Page Config ---
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Functions ---
@st.cache_resource
def load_best_model():
    """Load the best performing model"""
    if os.path.exists(BEST_MODEL_PATH):
        return joblib.load(BEST_MODEL_PATH)
    return None

@st.cache_data
def load_model_comparison():
    """Load model comparison results"""
    if os.path.exists(COMPARISON_REPORT_PATH):
        return pd.read_csv(COMPARISON_REPORT_PATH)
    return None

# Load model and comparison data
best_model = load_best_model()
comparison_df = load_model_comparison()

# Safe score formatting
if comparison_df is not None and len(comparison_df) > 0:
    best_score = comparison_df.iloc[0]["Test_ROC_AUC"]
    best_score_formatted = f"{best_score:.4f}"
    best_model_name = comparison_df.iloc[0]["Model"]
else:
    best_score = None
    best_score_formatted = "N/A"
    best_model_name = "Advanced ML Model"

if best_model is None:
    st.error("⚠️ Model file could not be loaded. Please check the model path.")
    st.stop()

# --- Main Header ---
st.markdown("""
<div class="main-header">
    <h1>📊 Customer Churn Prediction Dashboard</h1>
    <p>Advanced Analytics Platform for Customer Retention</p>
</div>
""", unsafe_allow_html=True)

# --- Model Information Section ---
if comparison_df is not None:
    st.markdown(f"""
    <div class="best-model-banner">
        <h2>🥇 Best Model: {best_model_name}</h2>
        <h3>ROC AUC Score: {best_score_formatted}</h3>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🎯 About This Application")
    st.success(f"""
    **Best Model Performance:**
    - Model: {best_model_name}
    - ROC AUC: {best_score_formatted}
    
    This application uses the best performing model from our comprehensive MLflow comparison.
    """)
    
    st.markdown("### 📋 Feature Categories")
    with st.expander("👤 Personal Information"):
        st.write("""
        - **Gender**: Customer's gender
        - **Senior Citizen**: Age category (65+)
        - **Partner**: Has a partner/spouse
        - **Dependents**: Has dependent family members
        - **Tenure**: Months as customer
        - **Monthly Charges**: Monthly bill amount
        - **Contract**: Service agreement type
        - **Payment Method**: How customer pays
        - **Paperless Billing**: Electronic billing preference
        """)
    
    with st.expander("🌐 Internet Services"):
        st.write("""
        - **Internet Service**: Type of internet connection
        - **Online Security**: Security service subscription
        - **Online Backup**: Backup service subscription
        - **Device Protection**: Device protection plan
        - **Tech Support**: Technical support subscription
        - **Streaming TV**: TV streaming service
        - **Streaming Movies**: Movie streaming service
        """)
    
    with st.expander("📞 Phone Services"):
        st.write("""
        - **Phone Service**: Basic phone service
        - **Multiple Lines**: Multiple phone lines
        """)

# --- Input Form ---
st.markdown("""
<div class="section-header">
    <h3>📝 Customer Information Input</h3>
</div>
""", unsafe_allow_html=True)

# Create three columns for input sections
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("#### 👤 Personal Information")
    gender = st.selectbox("👤 Gender", ["Male", "Female"], help="Customer's gender")
    senior_citizen = st.selectbox(
        "👴 Senior Citizen", 
        [0, 1], 
        format_func=lambda x: "Yes (65+)" if x==1 else "No (<65)",
        help="Whether customer is 65 years or older"
    )
    partner = st.selectbox("💏 Partner", ["Yes", "No"], help="Does customer have a partner/spouse?")
    dependents = st.selectbox("👨‍👩‍👧‍👦 Dependents", ["Yes", "No"], help="Does customer have dependent family members?")
    tenure = st.number_input(
        "📅 Tenure (months)", 
        min_value=0, max_value=100, value=12,
        help="Number of months as a customer"
    )
    monthly_charges = st.number_input(
        "💰 Monthly Charges ($)", 
        min_value=0.0, max_value=500.0, value=70.0,
        help="Monthly bill amount"
    )

with col2:
    st.markdown("#### 🌐 Internet Services")
    internet_service = st.selectbox(
        "🌐 Internet Service", 
        ["DSL", "Fiber optic", "No"],
        help="Type of internet service"
    )
    
    # Auto-set dependent features if no internet service
    if internet_service == "No":
        online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"
        st.info("ℹ️ Internet-related services automatically set to 'No internet service'")
    else:
        online_security = st.selectbox(
            "🔒 Online Security", 
            ["Yes", "No", "No internet service"],
            help="Online security service subscription"
        )
        online_backup = st.selectbox(
            "💾 Online Backup", 
            ["Yes", "No", "No internet service"],
            help="Online backup service subscription"
        )
        device_protection = st.selectbox(
            "🛡️ Device Protection", 
            ["Yes", "No", "No internet service"],
            help="Device protection plan"
        )
        tech_support = st.selectbox(
            "🔧 Tech Support", 
            ["Yes", "No", "No internet service"],
            help="Technical support subscription"
        )
        streaming_tv = st.selectbox(
            "📺 Streaming TV", 
            ["Yes", "No", "No internet service"],
            help="TV streaming service"
        )
        streaming_movies = st.selectbox(
            "🎬 Streaming Movies", 
            ["Yes", "No", "No internet service"],
            help="Movie streaming service"
        )

with col3:
    st.markdown("#### 📞 Phone & Payment Services")
    phone_service = st.selectbox("📞 Phone Service", ["Yes", "No"], help="Basic phone service")
    if phone_service == "No":
        multiple_lines = "No phone service"
        st.info("ℹ️ Multiple Lines automatically set to 'No phone service'")
    else:
        multiple_lines = st.selectbox(
            "📱 Multiple Lines", 
            ["Yes", "No", "No phone service"],
            help="Multiple phone lines service"
        )
    
    st.markdown("#### 💳 Billing Information")
    contract = st.selectbox(
        "📋 Contract Type", 
        ["Month-to-month", "One year", "Two year"],
        help="Service agreement duration"
    )
    payment_method = st.selectbox(
        "💳 Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        help="How customer pays their bills"
    )
    paperless_billing = st.selectbox(
        "📄 Paperless Billing", 
        ["Yes", "No"],
        help="Electronic billing preference"
    )

# --- Input Summary ---
st.markdown("""
<div class="section-header">
    <h3>📋 Customer Profile Summary</h3>
</div>
""", unsafe_allow_html=True)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown(f"""
    <div class="input-summary">
        <h4>👤 Personal Profile</h4>
        <ul>
            <li><strong>Gender:</strong> {gender}</li>
            <li><strong>Senior Citizen:</strong> {"Yes" if senior_citizen == 1 else "No"}</li>
            <li><strong>Partner:</strong> {partner}</li>
            <li><strong>Dependents:</strong> {dependents}</li>
            <li><strong>Tenure:</strong> {tenure} months</li>
            <li><strong>Monthly Charges:</strong> ${monthly_charges:.2f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with summary_col2:
    st.markdown(f"""
    <div class="input-summary">
        <h4>🌐 Services Profile</h4>
        <ul>
            <li><strong>Internet:</strong> {internet_service}</li>
            <li><strong>Phone:</strong> {phone_service}</li>
            <li><strong>Multiple Lines:</strong> {multiple_lines}</li>
            <li><strong>Online Security:</strong> {online_security}</li>
            <li><strong>Streaming TV:</strong> {streaming_tv}</li>
            <li><strong>Streaming Movies:</strong> {streaming_movies}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with summary_col3:
    st.markdown(f"""
    <div class="input-summary">
        <h4>💳 Account Profile</h4>
        <ul>
            <li><strong>Contract:</strong> {contract}</li>
            <li><strong>Payment Method:</strong> {payment_method}</li>
            <li><strong>Paperless Billing:</strong> {paperless_billing}</li>
            <li><strong>Online Backup:</strong> {online_backup}</li>
            <li><strong>Device Protection:</strong> {device_protection}</li>
            <li><strong>Tech Support:</strong> {tech_support}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Prepare input DataFrame ---
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [monthly_charges * tenure]  # Calculate TotalCharges
})

# --- Prediction Section ---
st.markdown("""
<div class="section-header">
    <h3>🔮 Churn Prediction Analysis</h3>
</div>
""", unsafe_allow_html=True)

# Threshold setting
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    threshold = st.slider(
        "🎯 Set Churn Decision Threshold", 
        0.0, 1.0, 0.5, 0.01,
        help="Probability threshold above which a customer is classified as 'likely to churn'"
    )
    st.info(f"Current threshold: {threshold:.1%} - Customers with churn probability ≥ {threshold:.1%} will be classified as 'likely to churn'")

# Prediction button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🔮 Predict Customer Churn", type="primary", use_container_width=True)

if predict_button:
    try:
        pred_proba = best_model.predict_proba(input_data)[0][1]
        prediction = int(pred_proba >= threshold)
        
        st.markdown("### 📊 Prediction Results")
        
        # Create two columns for results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("#### 🎯 Churn Prediction")
            if prediction == 1:
                st.markdown("""
                <div class="churn-yes">
                    ⚠️ Customer Likely to CHURN<br>
                    
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="churn-no">
                    ✅ Customer Likely to STAY<br>
                    
                </div>
                """, unsafe_allow_html=True)
        
        with result_col2:
            st.markdown("#### 📈 Churn Probability")
            
            # Determine risk category based on threshold
            if pred_proba >= threshold:
                risk_category = "High"
                gauge_colors = {
                    'low': "#28a745",    # Green
                    'medium': "#ffc107", # Yellow
                    'high': "#dc3545"    # Red
                }
            else:
                half_threshold = threshold / 2
                if pred_proba < half_threshold:
                    risk_category = "Low"
                else:
                    risk_category = "Medium"
                gauge_colors = {
                    'low': "#28a745",    # Green
                    'medium': "#ffc107", # Yellow  
                    'high': "#dc3545"    # Red
                }
            
            # Create a gauge chart using plotly with dynamic colors
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk %", 'font': {'size': 18, 'color': '#2c3e50'}},
                number = {'suffix': "%", 'font': {'size': 32, 'color': 'white', 'family': 'Arial Black'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "#2c3e50", 'tickfont': {'size': 14, 'color': '#2c3e50'}},
                    'bar': {'color': "#667eea", 'thickness': 0.3},
                    'steps': [
                        {'range': [0, threshold * 50], 'color': gauge_colors['low']},      # Low risk
                        {'range': [threshold * 50, threshold * 100], 'color': gauge_colors['medium']},  # Medium risk
                        {'range': [threshold * 100, 100], 'color': gauge_colors['high']}   # High risk
                    ],
                    'threshold': {
                        'line': {'color': "#2c3e50", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig.update_layout(
                height=300,
                font={'color': "#2c3e50"},
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        st.markdown("### 📋 Detailed Analysis")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4>📊 Churn Probability</h4>
                <h2 style="color: #667eea;">{pred_proba:.1%}</h2>
                <p>Customer's likelihood to churn</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            confidence = abs(pred_proba - 0.5) * 2
            st.markdown(f"""
            <div class="metric-container">
                <h4>🎯 Model Confidence</h4>
                <h2 style="color: #667eea;">{confidence:.1%}</h2>
                <p>Prediction confidence level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            # Calculate risk thresholds
            half_threshold = threshold / 2
            
            # Updated risk category logic based on threshold
            if pred_proba >= threshold:
                risk_category = "High"
                risk_color = "#dc3545"
            elif pred_proba >= half_threshold:
                risk_category = "Medium"
                risk_color = "#ffc107"
            else:
                risk_category = "Low"
                risk_color = "#28a745"
                
            st.markdown(f"""
            <div class="metric-container">
                <h4>⚠️ Risk Category</h4>
                <h2 style="color: {risk_color};">{risk_category}</h2>
                <p>Churn risk level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            retention_score = (1 - pred_proba) * 100
            st.markdown(f"""
            <div class="metric-container">
                <h4>💚 Retention Score</h4>
                <h2 style="color: #667eea;">{retention_score:.0f}/100</h2>
                <p>Customer loyalty index</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 Recommended Actions")
        
        # Calculate risk category for recommendations (same logic as above)
        half_threshold = threshold / 2
        if pred_proba >= threshold:
            risk_category = "High"
        elif pred_proba >= half_threshold:
            risk_category = "Medium"
        else:
            risk_category = "Low"
        
        if risk_category == "High":
            st.error("""
            **🚨 High Churn Risk - Immediate Action Required:**
            
            1. **Priority Contact**: Reach out to customer within 24-48 hours
            2. **Retention Offer**: Consider special discounts or service upgrades  
            3. **Service Review**: Analyze their service usage and satisfaction
            4. **Personal Touch**: Assign a dedicated account manager
            5. **Feedback Collection**: Conduct exit interview to understand concerns
            """)
        elif risk_category == "Medium":
            st.warning("""
            **⚠️ Medium Churn Risk - Proactive Engagement Needed:**
            
            1. **Customer Outreach**: Schedule a satisfaction call within 1-2 weeks
            2. **Service Enhancement**: Offer additional services or upgrades
            3. **Usage Analysis**: Review their service patterns for optimization
            4. **Loyalty Program**: Enroll in customer loyalty rewards
            5. **Regular Monitoring**: Track engagement levels closely
            """)
        else:
            st.success("""
            **✅ Low Churn Risk - Maintain Engagement:**
            
            1. **Customer Appreciation**: Send thank you notes or loyalty rewards
            2. **Upsell Opportunities**: Introduce new services that match their profile
            3. **Regular Check-ins**: Periodic satisfaction surveys
            4. **Community Building**: Invite to customer events or programs
            5. **Service Optimization**: Ensure they're getting maximum value
            """)
            
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.info("Please ensure all fields are filled correctly and the model is properly loaded.")

# --- Model Performance Section (IMPROVED) ---
if comparison_df is not None:
    with st.expander("📈 Model Performance Metrics", expanded=False):
        best_model_metrics = comparison_df.iloc[0]
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("🎯 ROC AUC", best_score_formatted)
        with perf_col2:
            st.metric("✅ Accuracy", f"{best_model_metrics['Test_Accuracy']:.4f}")
        with perf_col3:
            st.metric("🎪 Precision", f"{best_model_metrics['Test_Precision']:.4f}")
        with perf_col4:
            st.metric("🔄 Recall", f"{best_model_metrics['Test_Recall']:.4f}")
        
        # Model comparison chart
        if len(comparison_df) > 1:
            st.markdown("#### 🏆 Model Comparison")
            fig = px.bar(
                comparison_df.head(5), 
                x="Model", 
                y="Test_ROC_AUC",
                title="Top 5 Models Performance Comparison",
                color="Test_ROC_AUC",
                color_continuous_scale="viridis"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.dataframe(
                comparison_df.head(5).style.highlight_max(axis=0, subset=["Test_ROC_AUC", "Test_Accuracy", "Test_F1"])
                                        .format({
                                            "Test_ROC_AUC": "{:.4f}",
                                            "Test_Accuracy": "{:.4f}",
                                            "Test_Precision": "{:.4f}",
                                            "Test_Recall": "{:.4f}",
                                            "Test_F1": "{:.4f}"
                                        }),
                use_container_width=True
            )

# --- Developer Section (Optional) ---
st.markdown("---")
show_dev_info = st.checkbox("🔧 Show Developer Information")

if show_dev_info:
    st.markdown("""
    <div class="developer-section">
        <h3>🔬 Developer Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    dev_col1, dev_col2 = st.columns(2)
    
    with dev_col1:
        st.markdown("#### 🏆 Model Details")
        if comparison_df is not None:
            st.write(f"**Best Model:** {best_model_name}")
            st.write(f"**ROC AUC Score:** {best_score_formatted}")
            st.write(f"**Model Path:** {BEST_MODEL_PATH}")
            st.write(f"**Report Path:** {COMPARISON_REPORT_PATH}")
        
    with dev_col2:
        st.markdown("#### 📊 System Status")
        st.write(f"**Model Loaded:** {'✅ Yes' if best_model is not None else '❌ No'}")
        st.write(f"**Comparison Data:** {'✅ Available' if comparison_df is not None else '❌ Missing'}")
        st.write(f"**Total Models Compared:** {len(comparison_df) if comparison_df is not None else 'N/A'}")

# --- Footer ---
st.markdown("---")

# Safe formatting for footer
if comparison_df is not None:
    model_name = comparison_df.iloc[0]["Model"] if len(comparison_df) > 0 else "Best ML Model"
    model_score = best_score_formatted
else:
    model_name = "Advanced ML Model" 
    model_score = "N/A"

st.markdown(f"""
<div style="text-align: center; color: #6c757d; margin-top: 2rem;">
    <p>🏆 <strong>Advanced Churn Prediction Dashboard</strong> | Powered by {model_name}</p>
    <p><small>💡 Built with MLflow model comparison • Streamlit • Advanced Machine Learning</small></p>
    <p><small>🎯 Model Performance: {model_score} ROC AUC Score</small></p>
</div>
""", unsafe_allow_html=True)