# 🌾 Agri-Food CO₂ Emission Analysis Project 🌍

This project focuses on analyzing and forecasting CO₂ emissions within the agri-food sector using machine learning techniques. It encompasses data preprocessing, exploratory data analysis (EDA), model development, and web deployment using Streamlit. Additionally, an interactive dashboard is built using Power BI to visualize key insights. 📈

# Table of Contents

 1.   **Introduction**

 2.   **Dataset Description**

 3.  **Machine Learning for CO₂ Emission Prediction**

 4.   **Web Application Deployment**

 5.   **Usage Instructions**

# Introduction 🌱

The agri-food sector is responsible for approximately 62% of global annual CO₂ emissions . This project aims to delve into the dataset to extract valuable insights and develop predictive models to forecast CO₂ emissions. Understanding these patterns is crucial for developing sustainable practices and mitigating climate change impacts.​
Kaggle

# Dataset Description 📄

The dataset comprises information related to CO₂ emissions in the agri-food sector. Key features include:​

**Country**: Name of the country.​

**Year**: Year of the recorded data.​

**CO₂ Emissions**: Amount of CO₂ emissions measured in metric tons.​

**Agricultural Activities**: Specific activities contributing to CO₂ emissions.​

**Other Relevant Features**: Variables such as land use, crop production, and livestock numbers.​

For a detailed exploration of the dataset, refer to the Kaggle dataset page .​

# Machine Learning for CO₂ Emission Prediction 🤖

**EDA and data cleaning**
1. Drop unnecessary columns
2. Handel missing values
3. Feature Engineering
4. Label encoding for categorical values
5. Scaling
6. Use Random forest regressor for predict the course price
7. Evaluate the model performance​


# Web Application Deployment🚀
### Uses a Streamlit licrary to deploy our machine learning model for prediction and to navigate our dataset using mulyiple pages:
1. *Predict*: a page to predict the course price using the machine learning model 
2. *Dataset Insights*: gain an insights about the dataet using description and summary statistics methods 
3. *Model Performance*: display the performance and metrices of our regression model

# Usage of web application
1. Ensure you install the requierments libraries exists in the requierments.txt file 
2. Run all python scripts in the 'src' folde
3. Ensure you run the "App.py" python script successfully
4. Open the terminal and run the following bash command:
``` Bash
    cd 'path\src'
    streamlit run App.py
```