import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Football Injury Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_models():
    try:
        with open(BASE_DIR / 'random_forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open(BASE_DIR / 'pca_object.pkl', 'rb') as file:
            pca = pickle.load(file)
        with open(BASE_DIR / 'model_columns.pkl', 'rb') as file:
            columns = pickle.load(file)
        return model, pca, columns
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure all .pkl files are in the same directory.")
        return None, None, None

model, pca, model_columns = load_models()

# Position mapping
POSITION_MAP = {'Defender': 0, 'Forward': 1, 'Goalkeeper': 2, 'Midfielder': 3}

# Sidebar navigation
st.sidebar.title("⚽ Navigation")
page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Analysis", "About"])

if model is not None:
    
    # ==================== SINGLE PREDICTION ====================
    if page == "Single Prediction":
        st.title("⚽ Football Player Injury Risk Predictor")
        st.markdown("Comprehensive injury risk assessment based on 18 key metrics")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 👤 Player Profile")
                player_name = st.text_input("Player Name (optional)", placeholder="e.g., Mohamed Salah")
                age = st.slider("Age", 18, 40, 25)
                position = st.selectbox("Position", list(POSITION_MAP.keys()))
                previous_injury_count = st.number_input("Previous Severe Injuries", 0, 10, 0)
                
                st.markdown("#### 📏 Physical Metrics")
                height = st.number_input("Height (cm)", 150.0, 210.0, 180.0, 0.1)
                weight = st.number_input("Weight (kg)", 50.0, 120.0, 75.0, 0.1)
                
                # Auto-calculate BMI
                bmi = weight / ((height / 100) ** 2)
                st.metric("BMI (Auto-calculated)", f"{bmi:.1f}")
                
            with col2:
                st.markdown("#### 🏃 Training & Performance")
                training_hours = st.number_input("Training Hours/Week", 0, 30, 15)
                matches_played = st.number_input("Matches Last Season", 0, 60, 30)
                warmup_adherence = st.slider("Warmup Adherence (%)", 0, 100, 90)
                
                st.markdown("#### 💪 Biomechanics")
                knee_strength = st.slider("Knee Strength (0-100)", 0, 100, 75)
                hamstring_flex = st.slider("Hamstring Flexibility (0-100)", 0, 100, 65)
                sprint_speed = st.number_input("Sprint Speed 10m (seconds)", 1.0, 3.0, 1.5, 0.01)
                
            with col3:
                st.markdown("#### ⚡ Agility & Coordination")
                agility_score = st.slider("Agility Score (0-100)", 0, 100, 85)
                balance_score = st.slider("Balance Score (0-100)", 0, 100, 80)
                reaction_time = st.number_input("Reaction Time (ms)", 100, 500, 250)
                
                st.markdown("#### 😴 Wellness")
                sleep_hours = st.number_input("Sleep Hours/Night", 5.0, 10.0, 7.5, 0.1)
                stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
                nutrition_quality = st.slider("Nutrition Quality (1-10)", 1, 10, 7)
            
            submit_button = st.form_submit_button("🔮 Predict Injury Risk", type="primary", use_container_width=True)
        
        if submit_button:
            # Gather input data in correct order
            input_data = {
                'Height_cm': height,
                'Weight_kg': weight,
                'Training_Hours_Per_Week': training_hours,
                'Matches_Played_Past_Season': matches_played,
                'Knee_Strength_Score': knee_strength,
                'Hamstring_Flexibility': hamstring_flex,
                'Reaction_Time_ms': reaction_time,
                'Balance_Test_Score': balance_score,
                'Sprint_Speed_10m_s': sprint_speed,
                'Agility_Score': agility_score,
                'Sleep_Hours_Per_Night': sleep_hours,
                'Stress_Level_Score': stress_level,
                'Nutrition_Quality_Score': nutrition_quality,
                'Warmup_Routine_Adherence': warmup_adherence,
                'BMI': bmi,
                'Age': age,
                'Position': POSITION_MAP[position],
                'Previous_Injury_Count': previous_injury_count
            }
            
            # Create DataFrame with correct column order
            input_df = pd.DataFrame([input_data])[model_columns]
            
            # Apply PCA and predict
            input_pca = pca.transform(input_df)
            prediction_proba = model.predict_proba(input_pca)[0][1]
            risk_percentage = round(prediction_proba * 100, 1)
            
            st.markdown("---")
            
            # Risk Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Injury Risk - {player_name if player_name else 'Player'}", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': '#90EE90'},
                        {'range': [25, 50], 'color': '#FFD700'},
                        {'range': [50, 75], 'color': '#FFA500'},
                        {'range': [75, 100], 'color': '#FF6B6B'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 75}
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk Category
            col1, col2, col3 = st.columns(3)
            with col1:
                if risk_percentage >= 75:
                    st.error("🚨 **CRITICAL RISK**")
                    risk_msg = "Immediate medical evaluation required"
                elif risk_percentage >= 50:
                    st.warning("⚠️ **HIGH RISK**")
                    risk_msg = "Significant preventive measures needed"
                elif risk_percentage >= 25:
                    st.info("🟡 **MODERATE RISK**")
                    risk_msg = "Monitor closely and adjust training"
                else:
                    st.success("✅ **LOW RISK**")
                    risk_msg = "Continue current practices"
            
            with col2:
                st.metric("Risk Score", f"{risk_percentage}%")
            with col3:
                st.metric("Status", risk_msg)
            
            # Risk Factors Analysis
            st.subheader("📊 Key Risk Factors")
            
            risk_factors = []
            
            # Training overload
            if training_hours > 20:
                risk_factors.append(('Training Overload', (training_hours - 20) * 5, 'Excessive training hours'))
            
            # Previous injuries
            if previous_injury_count > 0:
                risk_factors.append(('Injury History', previous_injury_count * 15, f'{previous_injury_count} previous injuries'))
            
            # Physical metrics
            if knee_strength < 70:
                risk_factors.append(('Knee Weakness', (70 - knee_strength), 'Below optimal strength'))
            
            if hamstring_flex < 60:
                risk_factors.append(('Poor Flexibility', (60 - hamstring_flex), 'Limited hamstring flexibility'))
            
            # Wellness factors
            if sleep_hours < 7:
                risk_factors.append(('Sleep Deficit', (7 - sleep_hours) * 10, 'Insufficient recovery time'))
            
            if stress_level > 6:
                risk_factors.append(('High Stress', (stress_level - 6) * 8, 'Elevated stress levels'))
            
            if warmup_adherence < 80:
                risk_factors.append(('Warmup Issues', (80 - warmup_adherence) / 2, 'Inconsistent warmup routine'))
            
            # Age factor
            if age > 30:
                risk_factors.append(('Age Factor', (age - 30) * 2, 'Age-related risk increase'))
            
            # Match fatigue
            if matches_played > 45:
                risk_factors.append(('Match Fatigue', (matches_played - 45), 'High match exposure'))
            
            if risk_factors:
                risk_factors.sort(key=lambda x: x[1], reverse=True)
                
                factor_names = [f[0] for f in risk_factors[:6]]
                factor_values = [f[1] for f in risk_factors[:6]]
                factor_desc = [f[2] for f in risk_factors[:6]]
                
                fig_factors = go.Figure(data=[
                    go.Bar(
                        y=factor_names,
                        x=factor_values,
                        orientation='h',
                        marker=dict(color=factor_values, colorscale='RdYlGn_r'),
                        text=[f"{v:.1f}" for v in factor_values],
                        textposition='auto',
                        hovertext=factor_desc,
                        hoverinfo='text'
                    )
                ])
                
                fig_factors.update_layout(
                    title="Top Contributing Risk Factors",
                    xaxis_title="Impact Score",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_factors, use_container_width=True)
            else:
                st.success("✅ No significant risk factors identified!")
            
            # Recommendations
            st.subheader("💡 Personalized Recommendations")
            
            recommendations = []
            
            if risk_percentage >= 75:
                recommendations.extend([
                    "🚨 **URGENT**: Schedule immediate medical evaluation",
                    "⏸️ Consider reducing training intensity by 30-40%",
                    "💊 Implement comprehensive recovery protocols",
                    "📊 Daily monitoring of physical condition required"
                ])
            elif risk_percentage >= 50:
                recommendations.extend([
                    "⚠️ Reduce training load and monitor fatigue",
                    "🏥 Schedule preventive physiotherapy sessions",
                    "💪 Focus on strength and conditioning"
                ])
            
            if training_hours > 20:
                recommendations.append("⏱️ Reduce training hours - current load is excessive")
            
            if previous_injury_count >= 2:
                recommendations.append("🏥 Regular medical checkups due to injury history")
            
            if knee_strength < 70 or hamstring_flex < 60:
                recommendations.append("💪 Prioritize strength and flexibility training")
            
            if sleep_hours < 7:
                recommendations.append("😴 Increase sleep duration to at least 7-8 hours")
            
            if stress_level > 6:
                recommendations.append("🧘 Implement stress management techniques")
            
            if warmup_adherence < 80:
                recommendations.append("🏃 Improve warmup routine consistency")
            
            if not recommendations:
                recommendations.append("✅ Maintain current training and wellness practices")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
    
    # ==================== BATCH ANALYSIS ====================
    elif page == "Batch Analysis":
        st.title("📁 Batch Player Analysis")
        
        with st.expander("📥 Download CSV Template"):
            st.markdown("**Required columns (in any order):**")
            st.markdown(", ".join(model_columns))
            
            sample_data = pd.DataFrame({
                'Height_cm': [180, 175, 185],
                'Weight_kg': [75, 70, 80],
                'Training_Hours_Per_Week': [15, 20, 18],
                'Matches_Played_Past_Season': [30, 45, 25],
                'Knee_Strength_Score': [75, 65, 80],
                'Hamstring_Flexibility': [65, 60, 70],
                'Reaction_Time_ms': [250, 280, 230],
                'Balance_Test_Score': [80, 75, 85],
                'Sprint_Speed_10m_s': [1.5, 1.6, 1.4],
                'Agility_Score': [85, 80, 90],
                'Sleep_Hours_Per_Night': [7.5, 7.0, 8.0],
                'Stress_Level_Score': [5, 6, 4],
                'Nutrition_Quality_Score': [7, 6, 8],
                'Warmup_Routine_Adherence': [90, 85, 95],
                'BMI': [23.1, 22.9, 23.4],
                'Age': [25, 28, 22],
                'Position': [0, 1, 3],
                'Previous_Injury_Count': [1, 3, 0]
            })
            st.dataframe(sample_data)
            csv = sample_data.to_csv(index=False)
            st.download_button("Download Template", csv, "template.csv", "text/csv")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file and st.button("⚡ Analyze All Players", type="primary"):
            try:
                batch_df = pd.read_csv(uploaded_file)
                batch_input = batch_df[model_columns]
                batch_pca = pca.transform(batch_input)
                batch_proba = model.predict_proba(batch_pca)[:, 1]
                batch_percent = np.round(batch_proba * 100, 1)
                
                results = batch_df.copy()
                results["Risk_%"] = batch_percent
                results["Risk_Level"] = pd.cut(batch_percent, bins=[0, 25, 50, 75, 100], 
                                              labels=["Low", "Moderate", "High", "Critical"])
                
                # Summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Players", len(results))
                with col2:
                    st.metric("Avg Risk", f"{results['Risk_%'].mean():.1f}%")
                with col3:
                    critical = len(results[results["Risk_%"] >= 75])
                    st.metric("Critical Risk", critical)
                with col4:
                    low = len(results[results["Risk_%"] < 25])
                    st.metric("Low Risk", low)
                
                # Distribution chart
                fig = px.histogram(results, x="Risk_%", nbins=20, title="Team Risk Distribution")
                fig.add_vline(x=25, line_dash="dash", line_color="yellow")
                fig.add_vline(x=50, line_dash="dash", line_color="orange")
                fig.add_vline(x=75, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                results_sorted = results.sort_values("Risk_%", ascending=False)
                st.dataframe(results_sorted, use_container_width=True)
                
                # Download
                csv = results_sorted.to_csv(index=False)
                st.download_button("📥 Download Results", csv, "injury_risk_report.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ==================== ABOUT ====================
    elif page == "About":
        st.title("ℹ️ About This Application")
        st.markdown("""
        ### 🎯 Purpose
        Advanced ML-based injury risk prediction using 18 comprehensive player metrics.
        
        ### 🧠 Model Features (18 total)
        **Physical:** Height, Weight, BMI  
        **Training:** Hours/week, Matches played, Warmup adherence  
        **Strength:** Knee strength, Hamstring flexibility  
        **Performance:** Sprint speed, Agility, Balance, Reaction time  
        **Wellness:** Sleep, Stress, Nutrition  
        **Profile:** Age, Position, Injury history
        
        ### 📊 Risk Categories
        - 🟢 **Low (0-25%)**: Normal training  
        - 🟡 **Moderate (25-50%)**: Monitor closely  
        - 🟠 **High (50-75%)**: Preventive action needed  
        - 🔴 **Critical (75-100%)**: Immediate intervention
        
        ### 🔧 Technology
        Random Forest Classifier with PCA dimensionality reduction
        """)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: All 18 metrics are required for accurate predictions!")