import streamlit as st
import pandas as pd
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# PAGE CONFIG
st.set_page_config(page_title="Penguins Classifier", page_icon="🐧", layout="wide")

st.title("🐧 Penguins Classifier")
st.write("Predict penguin species using 4 numeric features.")


# LOAD DATA 
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv"
    )
    return df


df = load_data()


# TRAIN MODELS
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

    # Models
    knn = KNeighborsClassifier(n_neighbors=5)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)

    models = {
        "KNN": knn,
        "Decision Tree": dt,
    }

    metrics = []
    trained_models = {}
    preds_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        metrics.append({"Model": name, "Accuracy": acc})
        trained_models[name] = model
        preds_dict[name] = y_pred

    metrics_df = pd.DataFrame(metrics).sort_values(
        by="Accuracy", ascending=False
    ).reset_index(drop=True)

    best_model_name = metrics_df.iloc[0]["Model"]
    class_names = sorted(y.unique())

    return trained_models, metrics_df, best_model_name, y_test, preds_dict, class_names


models, metrics_df, best_model_name, y_test, preds_dict, class_names = train_models(df)

# TABS 
tab_data, tab_viz, tab_models, tab_pred = st.tabs(
    ["📘 Data", "📊 Visualization", "🤖 Models", "🔮 Prediction"]
)

# TAB: DATA
with tab_data:
    st.subheader("Dataset Preview")
    st.dataframe(df)

# TAB: VISUALIZATION
with tab_viz:

    st.subheader("Scatter Plot: Bill Length vs Body Mass")

    st.scatter_chart(
        df,
        x="bill_length_mm",
        y="body_mass_g",
        color="species",
    )

    # ----------------------------------------------------
    st.markdown("### Custom Scatter Plot")
    # ----------------------------------------------------

    numeric_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    x_feat = st.selectbox("X-axis", numeric_cols, index=0, key="x_feat")
    y_feat = st.selectbox("Y-axis", numeric_cols, index=1, key="y_feat")

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

    # ----------------------------------------------------
    st.markdown("### Feature Distribution (Histogram)")
    # ----------------------------------------------------

    hist_feat = st.selectbox("Select numeric feature:", numeric_cols, key="hist_feat")

    hist_chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(hist_feat, bin=alt.Bin(maxbins=30)),
            y="count()",
            tooltip=[hist_feat, "count()"],
        )
    )
    st.altair_chart(hist_chart, use_container_width=True)

    # ----------------------------------------------------
    st.markdown("### Summary Statistics")
    # ----------------------------------------------------
    st.dataframe(df.describe())

    # ----------------------------------------------------
    st.markdown("### Class Distributions")
    # ----------------------------------------------------

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


# TAB: MODELS
with tab_models:
    st.subheader("Model Accuracy")
    st.dataframe(metrics_df.style.format({"Accuracy": "{:.3f}"}))

    best_row = metrics_df.iloc[0]
    st.success(
        f"Best model: {best_row['Model']} (Accuracy = {best_row['Accuracy']:.3f})"
    )

    st.subheader("Confusion Matrix")

    cm_model_name = st.selectbox(
        "Choose model for confusion matrix",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )

    y_pred_cm = preds_dict[cm_model_name]
    cm = confusion_matrix(y_test, y_pred_cm, labels=class_names)

    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {c}" for c in class_names],
        columns=[f"Pred: {c}" for c in class_names],
    )

    st.write(f"Confusion matrix for **{cm_model_name}**:")
    st.dataframe(cm_df)


# TAB: PREDICTION
with tab_pred:
    st.subheader("Input Features")

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

    # Input display
    st.subheader("Your Input")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Bill Length (mm)", round(bill_length_mm, 2))
        st.metric("Flipper Length (mm)", round(flipper_length_mm, 2))

    with c2:
        st.metric("Bill Depth (mm)", round(bill_depth_mm, 2))
        st.metric("Body Mass (g)", round(body_mass_g, 2))


    user_input = {
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
    }

    # Model selection
    selected_model_name = st.selectbox(
        "Choose a model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )

    selected_model = models[selected_model_name]

    if st.button("Predict Penguin Species"):
        user_df = pd.DataFrame([user_input])
        pred = selected_model.predict(user_df)[0]
        proba = selected_model.predict_proba(user_df)[0]

        st.success(f"Predicted species: {pred}")

        proba_df = pd.DataFrame([proba], columns=selected_model.classes_)
        st.write("Class probabilities:")
        st.dataframe(proba_df)
