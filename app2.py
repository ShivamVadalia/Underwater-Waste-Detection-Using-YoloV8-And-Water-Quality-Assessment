import streamlit as st
from pycaret.classification import *
import pandas as pd
test_df = pd.read_csv('C:/Users/Acer/Documents/Neural_Ocean/Notebooks_PyFiles/test_data/test_df')

# Define the features and their types
features = {
    'pH': float,
    'Iron': float,
    'Nitrate': float,
    'Chloride': float,
    'Lead': float,
    'Zinc': float,
    'Color': str,
    'Turbidity': float,
    'Fluoride': float,
    'Copper': float,
    'Odor': float,
    'Sulfate': float,
    'Chlorine': float,
    'Manganese': float,
    'Total Dissolved Solids': float,
}

# Define the target variable
target_variable = 'Target'

# Define the color options
color_options = ['Colorless', 'Faint Yellow', 'Light Yellow', 'Near Colorless', 'Yellow', 'NaN']

quality = []
# Create a Streamlit app
def app2():
    st.title('Water Potability Test Model')
    # Load the pretrained model
    model = load_model(
        'C:/Users/Acer/Documents/Neural_Ocean/Notebooks_PyFiles/models/Water_Quality_Assessment'
        '/xgboost_without_source_month')
    # Create input widgets for each feature
    inputs = {}
    col1, col2, col3 = st.columns(3)
    for i, feature in enumerate(features.items()):
        if feature[0] == 'Color':
            col = col1
        elif i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3
        with col:
            if feature[0] == 'Color':
                inputs[feature[0]] = st.selectbox(f'{feature[0]}', options=color_options)
            else:
                inputs[feature[0]] = st.number_input(f'{feature[0]}', value=0.0, step=0.1, format='%.1f',
                                                     key=feature[0])

    # Add two buttons aligned in the center
    col1, col2 = st.columns([1,2])
    with col1:
        if st.button('Predict'):
            data = pd.DataFrame(inputs, index=range(0, 1), columns=inputs.keys())
            target = predict_model(model, data=data)
            quality.append(target['prediction_label'][0])
            if target['prediction_label'][0] == 0:
                st.success('The Water is fit for drinking and also for irrigation purpose')
            else:
                st.error('The Water is not fit for drinking or for irrigation purpose')

    with col2:
        if st.button('Random Inputs Predict'):
            data = test_df.sample(n=1)
            data.drop(['Target'], axis=1, inplace=True)
            st.write(data)
            target = predict_model(model, data=data)
            quality.append(target['prediction_label'][data.index[0]])
            if target['prediction_label'][data.index[0]]== 0:
                st.success('The Water is fit for drinking and also for irrigation purpose')
            else:
                st.error('The Water is not fit for drinking or for irrigation purpose')