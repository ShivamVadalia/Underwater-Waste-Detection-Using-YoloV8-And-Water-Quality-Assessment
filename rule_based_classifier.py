import streamlit as st
import pandas as pd


def is_habitable(pH, Iron, Nitrate, Chloride, Lead, Zinc, Turbidity, Fluoride, Copper, Sulfate, Chlorine, Manganese,
                 Total_Dissolved_Solids):
    if pH >= 6.5 and pH <= 9.0 and Iron < 0.3 and Nitrate < 10 and Chloride < 250 and Lead < 0.015 and Zinc < 5 and Turbidity < 5 and Fluoride >= 0.7 and Fluoride <= 1.5 and Copper < 1.3 and Sulfate < 250 and Chlorine < 4.0 and Manganese < 0.05 and Total_Dissolved_Solids < 500:
        return 0
    else:
        return 1


test_df = pd.read_csv('C:/Users/Acer/Documents/Neural_Ocean/Notebooks_PyFiles/test_data/test_df')

# Define the features and their types
features = {
    'pH': float,
    'Iron': float,
    'Nitrate': float,
    'Chloride': float,
    'Lead': float,
    'Zinc': float,
    'Turbidity': float,
    'Fluoride': float,
    'Copper': float,
    'Sulfate': float,
    'Chlorine': float,
    'Manganese': float,
    'Total Dissolved Solids': float,
}

quality_aquatic = []

def rbc():
    st.title('Water Quality Assessment Test')
    inputs = {}
    col1, col2, col3 = st.columns(3)
    for i, feature in enumerate(features.items()):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3
        with col:
                inputs[feature[0]] = st.number_input(f'{feature[0]}', value=0.0, step=0.1, format='%.1f',
                                                     key=feature[0])

    # Add two buttons aligned in the center
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button('Predict'):
            inputs_list = list(inputs.values())
            is_good = is_habitable(*inputs_list)
            if is_good == 0:
                st.success("Water quality is habitable for aquatic life")
            else:
                st.error("Water quality is not habitable for aquatic life")
            quality_aquatic.append(is_good)

    with col2:
        if st.button('Random Inputs Predict'):
            data = test_df.sample(n=1)
            data.drop(['Target', 'Color', 'Odor'], axis=1, inplace=True)
            st.write(data)
            # st.write(data.values.tolist())
            is_good = is_habitable(*data.values.tolist()[0])
            if is_good == 0:
                st.success("Water quality is habitable for aquatic life")
            else:
                st.error("Water quality is not habitable for aquatic life")
            quality_aquatic.append(is_good)
