import streamlit as st
import pandas as pd
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Penguins Classifier", page_icon="🐧", layout="wide")

st.title("🐧 Penguins Classifier")
st.write("Predict the penguin species using **4 numeric features**.")


# ---------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv"
    )
    return df


df = load_data()


# ---------------- TRAIN MODELS -----------------
@st.cache_resource
def train_models(df):
    feature_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    target_col = "species"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base Models
    knn = KNeighborsClassifier(n_neighbors=5)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)

    # Ensemble Model
    ensemble = VotingClassifier(
        estimators=[("knn", knn), ("dt", dt)],
        voting="hard",
    )

    models = {
        "KNN": knn,
        "Decision Tree": dt,
        "Ensemble (KNN + DT)": ensemble,
    }

    metrics = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        metrics.append({"Model": name, "Accuracy": acc})
        trained_models[name] = model

    metrics_df = pd.DataFrame(metrics).sort_values(
        by="Accuracy", ascending=False
    ).reset_index(drop=True)

    best_model_name = metrics_df.iloc[0]["Model"]

    return trained_models, metrics_df, best_model_name


models, metrics_df, best_model_name = train_models(df)


# ---------------- TABS -----------------
tab_data, tab_viz, tab_model, tab_pred = st.tabs(
    ["📘 Data", "📊 Visualization", "🤖 Models", "🔮 Prediction"]
)


# ---------------- TAB 1: DATA -----------------
with tab_data:
    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Class Distributions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Species:")
        st.bar_chart(df["species"].value_counts())

    with col2:
        st.write("Island:")
        st.bar_chart(df["island"].value_counts())

    with col3:
        st.write("Gender:")
        st.bar_chart(df["sex"].value_counts())


# ---------------- TAB 2: VISUALIZATION -----------------
with tab_viz:
    st.subheader("Scatter Plot: Bill Length vs Body Mass")

    st.scatter_chart(
        df,
        x="bill_length_mm",
        y="body_mass_g",
        color="species",
    )

    st.subheader("Custom Scatter Plot")

    numeric_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    x_feat = st.selectbox("X-axis", numeric_cols, index=0)
    y_feat = st.selectbox("Y-axis", numeric_cols, index=1)

    chart = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=x_feat,
            y=y_feat,
            color="species",
            tooltip=["species", x_feat, y_feat],
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


# ---------------- TAB 3: MODEL PERFORMANCE -----------------
with tab_model:
    st.subheader("Model Accuracy (using only 4 numeric features)")
    st.dataframe(metrics_df.style.format({"Accuracy": "{:.3f}"}))

    best_row = metrics_df.iloc[0]
    st.success(
        f"Best model: {best_row['Model']} (Accuracy = {best_row['Accuracy']:.3f})"
    )


# ---------------- TAB 4: PREDICTION -----------------
with tab_pred:
    st.subheader("Input Features")

    # Sidebar sliders for input
    bill_length_mm = st.slider(
        "Bill length (mm)",
        float(df.bill_length_mm.min()),
        float(df.bill_length_mm.max()),
        float(df.bill_length_mm.mean()),
    )

    bill_depth_mm = st.slider(
        "Bill depth (mm)",
        float(df.bill_depth_mm.min()),
        float(df.bill_depth_mm.max()),
        float(df.bill_depth_mm.mean()),
    )

    flipper_length_mm = st.slider(
        "Flipper length (mm)",
        float(df.flipper_length_mm.min()),
        float(df.flipper_length_mm.max()),
        float(df.flipper_length_mm.mean()),
    )

    body_mass_g = st.slider(
        "Body mass (g)",
        int(df.body_mass_g.min()),
        int(df.body_mass_g.max()),
        int(df.body_mass_g.mean()),
    )

    user_input = {
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
    }

    st.write("Your input:")
    st.json(user_input)

    # Model selection
    selected_model_name = st.selectbox(
        "Choose a model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )

    selected_model = models[selected_model_name]

    # Prediction button
    if st.button("Predict Penguin Species"):
        user_df = pd.DataFrame([user_input])
        pred = selected_model.predict(user_df)[0]
        proba = selected_model.predict_proba(user_df)[0]

        st.success(f"Predicted species: {pred}")

        proba_df = pd.DataFrame([proba], columns=selected_model.classes_)
        st.write("Class probabilities:")
        st.dataframe(proba_df)
