import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# ============ Page Configuration ============
st.set_page_config(layout="wide", page_title="Agri-Food CO₂ Emissions Dashboard")

# ============ Load Assets ============
# Logo
logo_path = r'C:\Users\Right Click\Desktop\ml_2\SRC\Co2.jpeg'
try:
    logo = Image.open(logo_path)
except Exception as e:
    logo = None

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\Right Click\Desktop\ml_2\DataSet\cleaned_data.csv')

data = load_data()

# Load Model and Scaler
model_path = r"C:\Users\Right Click\Desktop\ml_2\Work Space\model.pkl"
scaler_path = r"C:\Users\Right Click\Desktop\ml_2\Work Space\scaler.pkl"

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# ============ Apply Style ============
st.markdown(
    """
    <style>
    .main {
        background-color: #e3f9e5;
        padding: 2rem;
    }
    .css-1d391kg, .css-1lcbmhc {
        background-color: #b7f2bb !important;
    }
    h1, h2, h3, h4 {
        color: #2d6a4f;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #b7d5c4;
    }
    .stButton > button {
        background-color: #40916c;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #2d6a4f;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============ Sidebar ============
st.sidebar.title("Agri-Food CO₂ Emissions Dashboard")
page = st.sidebar.radio("Select a Page:", 
                       ["Home", "Correlation Heatmap", "Predict Emissions", "Global Emissions Map"])

# ============ Home Page ============
if page == "Home":
    st.title("Agri-Food CO₂ Emissions and Climate Change")
    if logo:
        st.image(logo, width=250)

    st.markdown("""
    This dashboard presents emissions from agri-food systems and their impact on climate change.

    ---

    ### Project Introduction

    The agri-food sector plays a critical role in greenhouse gas emissions.
    This project leverages data from FAO and IPCC to:
    - Explore and visualize emissions data.
    - Understand correlations between activities and emissions.
    - Predict total CO₂ emissions based on agri-food activities.

    We focus on emissions caused by farming, food processing, transportation, packaging, and population-related factors.
    """)
# ============ Correlation Heatmap Page ============
elif page == "Correlation Heatmap":
    st.header("Correlation Heatmap")
    st.markdown("Understanding how features relate to CO₂ Emissions")

    # Load cleaned data
    data = pd.read_csv(r'C:\Users\Right Click\Desktop\ml_2\DataSet\cleaned_data.csv')

    # Keep only important features
    selected_features = [
        'Urban population', 'Agrifood Systems Waste Disposal',
        'Food Household Consumption', 'IPPU', 'Food Packaging',
        'Manure applied to Soils', 'Crop Residues', 'Total_Population',
        'Fertilizers Manufacturing', 'Food Processing', 'total_emission'
    ]

    df = data[selected_features]

    # Correlation Heatmap for selected features
    correlation = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation, cmap="RdYlGn", annot=True, ax=ax)
    st.subheader("Correlation Heatmap (Top 10 features + target)")
    st.pyplot(fig)

    # Pie Chart - top 10 contributors (based on correlation)
    st.subheader("Top 10 Feature Contributions to CO₂ Emissions")

    top_features = correlation["total_emission"].abs().sort_values(ascending=False)[1:11]
    fig2, ax2 = plt.subplots()
    ax2.pie(top_features.values, labels=top_features.index, autopct="%1.1f%%", startangle=140)
    ax2.axis("equal")
    st.pyplot(fig2)

    # Bar Plot - strength of correlation with target
    st.subheader("Feature Correlation Strength with CO₂ Emissions")
    fig3, ax3 = plt.subplots()
    sns.barplot(x=top_features.values, y=top_features.index, palette="YlOrRd", ax=ax3)
    ax3.set_xlabel("Correlation Coefficient")
    ax3.set_xlim(0, 1)
    st.pyplot(fig3)

# ============ Prediction Page ============
# ============ Prediction Page ============
elif page == "Predict Emissions":
    import random  # استيراد مكتبة العشوائية
    st.header("Predict Total CO₂ Emissions")

    # اختيار الدولة
    st.subheader("Select a Country")
    selected_country = st.selectbox("Choose a Country", sorted(data['Area'].unique()))

    input_features = ['Savanna fires', 'Forest fires', 'Crop Residues', 'Rice Cultivation',
                      'Drained organic soils (CO2)', 'Pesticides Manufacturing',
                      'Food Transport', 'Forestland', 'Net Forest conversion',
                      'Food Household Consumption', 'Food Retail', 'On-farm Electricity Use',
                      'Food Packaging', 'Agrifood Systems Waste Disposal', 'Food Processing',
                      'Fertilizers Manufacturing', 'IPPU', 'Manure applied to Soils',
                      'Manure left on Pasture', 'Manure Management', 'Fires in organic soils',
                      'Fires in humid tropical forests', 'On-farm energy use',
                      'Rural population', 'Urban population', 'Total_Population']

    # إذا كانت البيانات محفوظة مسبقاً في session_state، استخدمها
    if 'user_input' in st.session_state:
        user_input = st.session_state['user_input']
    else:
        # إنشاء قيم افتراضية عشوائية لكل ميزة
        user_input = {feature: round(random.uniform(10, 1000), 2) for feature in input_features}

    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(input_features):
        with [col1, col2, col3][i % 3]:
            user_input[feature] = st.number_input(
                f"{feature}", 
                value=user_input.get(feature, round(random.uniform(10, 1000), 2)),  # قيمة افتراضية
                placeholder="Enter value"
            )

    if st.button("Predict CO₂ Emissions"):
        try:
            # تحقق من ملء جميع الحقول
            if any(v is None for v in user_input.values()):
                st.error("Please enter all input values before predicting.")
            else:
                input_df = pd.DataFrame([user_input])
                input_df = input_df[input_features]

                # تحقق من صلاحية الـ scaler
                if not hasattr(scaler, 'transform'):
                    raise TypeError(f"Scaler object is not valid. Got type: {type(scaler)}")

                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)

                st.success(f"Predicted Total CO₂ Emissions for {selected_country}: {prediction[0]:,.2f} kilotons")

                # بعد ما يتم التنبؤ
                avg_temp = data[data['Area'] == selected_country]['Average Temperature °C'].values
                avg_temp = avg_temp[0] if len(avg_temp) > 0 else 0

                # حفظ البيانات المتوقعة في session_state
                st.session_state['predicted_data'] = {
                    'Area': selected_country,
                    'total_emission': float(prediction[0]),
                    'Average Temperature °C': avg_temp
                }

                # حفظ المدخلات في session_state
                st.session_state['user_input'] = user_input

                # عرض المدخلات
                with st.expander("View Input Summary"):
                    st.dataframe(input_df.T.rename(columns={0: "Value"}))

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


# # ============ Map Visualization Page ============
elif page == "Global Emissions Map":
    st.header("Global Emissions Map")

    # التحقق من وجود بيانات تنبؤ في session_state
    if 'predicted_data' in st.session_state:
        country_data = pd.DataFrame([st.session_state['predicted_data']])
        selected_country = st.session_state['predicted_data']['Area']
        st.success(f"Showing prediction result for {selected_country}")
    else:
        st.warning("No prediction data found. Please predict emissions first.")
        st.stop()

    # تحضير الخريطة
    fig_map = px.scatter_geo(
        country_data,
        locations="Area",
        locationmode="country names",
        size="total_emission",
        hover_name="Area",
        hover_data=["Average Temperature °C"],
        color="total_emission",
        color_continuous_scale="RdYlGn_r",
        projection="natural earth",
        title=f"CO₂ Emissions Prediction for {selected_country}"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader(f"Recommendations for {selected_country}")

    emissions = country_data['total_emission'].iloc[0]

    if emissions > 100000:
        st.warning("""
        - **High emissions detected!**
        - **Recommendations:**
          - Implement large-scale sustainable agricultural practices
          - Reduce food waste in supply chains
          - Transition to renewable energy sources
          - Improve waste disposal management
          - Invest in carbon capture and storage technologies
        """)
    elif emissions > 50000:
        st.warning("""
        - **Moderate to High emissions detected.**
        - **Recommendations:**
          - Promote low-carbon farming techniques
          - Improve public transportation and logistics
          - Encourage energy efficiency in food processing
          - Increase afforestation and reforestation efforts
        """)
    else:
        st.warning("""
        - **Moderate emissions detected.**
        - **Recommendations:**
          - Increase awareness about sustainable farming and food consumption
          - Improve energy efficiency in the agricultural sector
          - Foster community-based climate action programs
        """)


