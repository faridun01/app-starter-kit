import streamlit as st
import pandas as pd

st.title('🎈 App Name')

st.write('Hello world!')

with st.expander("DATA"):
  st.write('**OUR DATA**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

with st.expander("Visualization"):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

with st.sidebar:
    st.header('Input features')

    island = st.selectbox("Island", ('Torgersen', 'Dream', 'Biscoe'))

    bill_length_mm = st.slider("Bill length (mm)", 32.1, 59.6, 35.1)
    bill_depth_mm = st.slider("Bill depth (mm)", 13.1, 21.5, 16.3)
    flipper_length_mm = st.slider("Flipper length (mm)", 172.0, 231.0, 195.0)
    body_mass_g = st.slider("Body mass (g)", 2700, 6300, 4200)

    sex = st.selectbox("Gender", ("male", "female"))

