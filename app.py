import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Crop Yield Predictor",layout="wide",page_icon = 'climate1.png')
st.markdown(
        r"""
        <style> 
        MainMenu {visibility: hidden;}
        .stAppDeployButton {
                visibility: hidden;
            }
        footer{visibility: hidden;}
        </style>
        """, unsafe_allow_html=True
    )
st.image('climate2.jpg', use_container_width=True)
with open('random_regression_model.pkl', 'rb') as model1:
    rf_model = joblib.load(model1)
with open('linear_regression_model.pkl', 'rb') as model2:
    linear_model = joblib.load(model2)
with open('svm_regression_model.pkl', 'rb') as model3:
    svm_model = joblib.load(model3)
with open('gradient_regression_model.pkl', 'rb') as model4:
    gradient_model = joblib.load(model4)
# Function to predict crop yield based on user input
AREA = ['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia',
       'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
       'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil',
       'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon', 'Canada',
       'Central African Republic', 'Chile', 'Colombia', 'Croatia',
       'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
       'Eritrea', 'Estonia', 'Finland', 'France', 'Germany', 'Ghana',
       'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras',
       'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy',
       'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon',
       'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi',
       'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico',
       'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal',
       'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway',
       'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal',
       'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal',
       'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan',
       'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand',
       'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom',
       'Uruguay', 'Zambia', 'Zimbabwe']
ITEM= ['Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat',
       'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams']
##2
def predict_yield_dynamic(area, item, year, avg_rainfall, pesticides, avg_temp, model_name, encoder_columns):
    # Create a DataFrame for the input
    input_data = {
        "Year": [year],
        "average_rain_fall_mm_per_year": [avg_rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [avg_temp]
    }

    # Convert to DataFrame
    df_input = pd.DataFrame(input_data)

    # Create a DataFrame for the one-hot encoded columns (initialized to 0)
    encoded_data = {col: [0] for col in encoder_columns if col not in df_input.columns}
    
    # Add the area and item columns to the encoded data
    area_col = f"Area_{area}"
    item_col = f"Item_{item}"
    if area_col in encoder_columns:
        encoded_data[area_col] = [1]
    else:
        print(f"Warning: Area '{area}' not found in training data. Prediction may be less accurate.")
    
    if item_col in encoder_columns:
        encoded_data[item_col] = [1]
    else:
        print(f"Warning: Item '{item}' not found in training data. Prediction may be less accurate.")
    
    # Convert to DataFrame
    df_encoded = pd.DataFrame(encoded_data)
    
    # Concatenate the original input DataFrame with the encoded DataFrame
    df_input = pd.concat([df_input, df_encoded], axis=1)

    # Reindex to match the training data columns (excluding the target column 'hg/ha_yield')
    df_input = df_input.reindex(columns=encoder_columns, fill_value=0)

    # Predict using the trained model
    # 'Linear Regression','Random Forest Regression','Support Vector Regression','Gradient Boosting Regression'
    if model_name == "Linear Regression":
        prediction = linear_model.predict(df_input)
    elif model_name == "Random Forest Regression":
        prediction = rf_model.predict(df_input)
    elif model_name == "Support Vector Regression":
        prediction = svm_model.predict(df_input)
    elif model_name == "Gradient Boosting Regression":
        prediction = gradient_model.predict(df_input)
    return prediction[0]


# Streamlit UI
st.title('Crop Yield Prediction')

# User input fields
model_name = st.selectbox("Select Regression Model", ['Linear Regression','Random Forest Regression','Support Vector Regression','Gradient Boosting Regression'])
area = st.selectbox('Select Area (Country):', AREA)
item = st.selectbox('Select Crop:', ITEM)
year = st.number_input('Enter Year:', min_value=1900, max_value=2050, value=2015)
rain_fall = st.number_input('Average Rainfall (mm per year):', min_value=0, max_value=5000, value=800)
pesticides = st.number_input('Pesticides (tonnes):', min_value=0, max_value=400000, value=0)
avg_temp = st.number_input("Temprature (°C):", min_value=0.0, max_value=60.0, value=0.0, step=0.1)

# Predict button
if st.button('Predict Yield'):
    encoder_columns = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area_Albania', 'Area_Algeria', 'Area_Angola', 'Area_Argentina', 'Area_Armenia', 'Area_Australia', 'Area_Austria', 'Area_Azerbaijan', 'Area_Bahamas', 'Area_Bahrain', 'Area_Bangladesh', 'Area_Belarus', 'Area_Belgium', 'Area_Botswana', 'Area_Brazil', 'Area_Bulgaria', 'Area_Burkina Faso', 'Area_Burundi', 'Area_Cameroon', 'Area_Canada', 'Area_Central African Republic', 'Area_Chile', 'Area_Colombia', 'Area_Croatia', 'Area_Denmark', 'Area_Dominican Republic', 'Area_Ecuador', 'Area_Egypt', 'Area_El Salvador', 'Area_Eritrea', 'Area_Estonia', 'Area_Finland', 'Area_France', 'Area_Germany', 'Area_Ghana', 'Area_Greece', 'Area_Guatemala', 'Area_Guinea', 'Area_Guyana', 'Area_Haiti', 'Area_Honduras', 'Area_Hungary', 'Area_India', 'Area_Indonesia', 'Area_Iraq', 'Area_Ireland', 'Area_Italy', 'Area_Jamaica', 'Area_Japan', 'Area_Kazakhstan', 'Area_Kenya', 'Area_Latvia', 'Area_Lebanon', 'Area_Lesotho', 'Area_Libya', 'Area_Lithuania', 'Area_Madagascar', 'Area_Malawi', 'Area_Malaysia', 'Area_Mali', 'Area_Mauritania', 'Area_Mauritius', 'Area_Mexico', 'Area_Montenegro', 'Area_Morocco', 'Area_Mozambique', 'Area_Namibia', 'Area_Nepal', 'Area_Netherlands', 'Area_New Zealand', 'Area_Nicaragua', 'Area_Niger', 'Area_Norway', 'Area_Pakistan', 'Area_Papua New Guinea', 'Area_Peru', 'Area_Poland', 'Area_Portugal', 'Area_Qatar', 'Area_Romania', 'Area_Rwanda', 'Area_Saudi Arabia', 'Area_Senegal', 'Area_Slovenia', 'Area_South Africa', 'Area_Spain', 'Area_Sri Lanka', 'Area_Sudan', 'Area_Suriname', 'Area_Sweden', 'Area_Switzerland', 'Area_Tajikistan', 'Area_Thailand', 'Area_Tunisia', 'Area_Turkey', 'Area_Uganda', 'Area_Ukraine', 'Area_United Kingdom', 'Area_Uruguay', 'Area_Zambia', 'Area_Zimbabwe', 'Item_Cassava', 'Item_Maize', 'Item_Plantains and others', 'Item_Potatoes', 'Item_Rice, paddy', 'Item_Sorghum', 'Item_Soybeans', 'Item_Sweet potatoes', 'Item_Wheat', 'Item_Yams']

    prediction = predict_yield_dynamic(area, item, year, rain_fall, pesticides, avg_temp, model_name=model_name, encoder_columns=encoder_columns)
    st.write(f"Predicted Crop Yield (hg/ha): {prediction:.2f}")

