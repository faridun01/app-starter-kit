import streamlit as st
import pandas as pd

st.title('🎈 Hello ;)')
st.write("Let's do it")

# ----------- LOAD DATA -----------
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

# ----------- EXPANDER: DATA -----------
with st.expander("DATA"):
    st.write("**OUR DATA**")
    st.dataframe(df)

# ----------- EXPANDER: VISUALIZATION -----------
with st.expander("Visualization"):
    st.scatter_chart(
        data=df,
        x='bill_length_mm',
        y='body_mass_g',
        color='species'
    )

# ----------- SIDEBAR INPUTS -----------
st.sidebar.header("Input features")

island = st.sidebar.selectbox("Island", ('Torgersen', 'Dream', 'Biscoe'))

bill_length_mm = st.sidebar.slider(
    "Bill length (mm)",
    min_value=float(df.bill_length_mm.min()),
    max_value=float(df.bill_length_mm.max()),
    value=float(df.bill_length_mm.mean())
)

bill_depth_mm = st.sidebar.slider(
    "Bill depth (mm)",
    min_value=float(df.bill_depth_mm.min()),
    max_value=float(df.bill_depth_mm.max()),
    value=float(df.bill_depth_mm.mean())
)

flipper_length_mm = st.sidebar.slider(
    "Flipper length (mm)",
    min_value=float(df.flipper_length_mm.min()),
    max_value=float(df.flipper_length_mm.max()),
    value=float(df.flipper_length_mm.mean())
)

body_mass_g = st.sidebar.slider(
    "Body mass (g)",
    min_value=int(df.body_mass_g.min()),
    max_value=int(df.body_mass_g.max()),
    value=int(df.body_mass_g.mean())
)

sex = st.sidebar.selectbox("Gender", ("male", "female"))

# ----------- SHOW SIDEBAR INPUT BACK TO USER -----------
user_input = {
    "island": island,
    "bill_length_mm": bill_length_mm,
    "bill_depth_mm": bill_depth_mm,
    "flipper_length_mm": flipper_length_mm,
    "body_mass_g": body_mass_g,
    "sex": sex
}

st.subheader("Your Input")
st.json(user_input)
