import streamlit as st
import pandas as pd
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# PAGE CONFIG
st.set_page_config(page_title="Penguins Classifier", page_icon="🐧", layout="wide")

st.title("🐧 Penguins Classifier")
st.write("Predict penguin species using numeric and categorical features.")

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
    # feature sets
    numeric_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    categorical_cols = ["island", "sex"]
    target_col = "species"

    X = df[numeric_cols + categorical_cols]
    y = df[target_col]

    # preprocessing: numeric passthrough, categorical one-hot
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # pipelines = preprocessing + model
    knn = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    dt = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]
    )

    rf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            )),
        ]
    )

    models = {
        "KNN": knn,
        "Decision Tree": dt,
        "Random Forest": rf,
    }

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
    st.markdown("### Class Distributions")
    # ----------------------------------------------------
    
    col1, col2, col3 = st.columns(3)
    
   # ----------------------------------------------------
st.markdown("### Class Distributions")
# ----------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Species Distribution:")

    species_df = df['species'].value_counts().reset_index()
    species_df.columns = ['species', 'count']

    species_chart = (
        alt.Chart(species_df)
        .mark_bar()
        .encode(
            x=alt.X('species:N', title="Species"),
            y=alt.Y('count:Q', title="Count"),
            color=alt.Color('species:N', title="Species")  # built-in categorical colors
        )
    )
    st.altair_chart(species_chart, use_container_width=True)

with col2:
    st.write("Island Distribution:")

    island_df = df['island'].value_counts().reset_index()
    island_df.columns = ['island', 'count']

    island_chart = (
        alt.Chart(island_df)
        .mark_bar()
        .encode(
            x=alt.X('island:N', title="Island"),
            y=alt.Y('count:Q', title="Count"),
            color=alt.Color('island:N', title="Island")  # built-in colors
        )
    )
    st.altair_chart(island_chart, use_container_width=True)

with col3:
    st.write("Gender Distribution:")

    gender_df = df['sex'].value_counts().reset_index()
    gender_df.columns = ['sex', 'count']

    gender_chart = (
        alt.Chart(gender_df)
        .mark_bar()
        .encode(
            x=alt.X('sex:N', title="Gender"),
            y=alt.Y('count:Q', title="Count"),
            color=alt.Color('sex:N', title="Gender")  # built-in colors
        )
    )
    st.altair_chart(gender_chart, use_container_width=True)



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

    # we need the same X_test that was used above; easiest is to recompute locally
    # but here we only stored predictions; confusion_matrix still works with y_test and y_pred_cm
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

    # numeric inputs
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

    # categorical inputs
    island = st.selectbox("Island", sorted(df["island"].unique()))
    sex = st.selectbox("Sex", sorted(df["sex"].unique()))

    # ---------------- DISPLAY USER INPUT ----------------
    st.subheader("Your Input")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Bill Length (mm)", round(bill_length_mm, 2))
        st.metric("Flipper Length (mm)", round(flipper_length_mm, 2))

    with c2:
        st.metric("Bill Depth (mm)", round(bill_depth_mm, 2))
        st.metric("Body Mass (g)", round(body_mass_g, 2))

    st.write(f"Island: {island}")
    st.write(f"Sex: {sex}")

    # Prepare input for model
    user_input = {
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "island": island,
        "sex": sex,
    }

    # ---------------- MODEL SELECTION ----------------
    st.subheader("Choose a Model")
    selected_model_name = st.selectbox(
        "",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )

    selected_model = models[selected_model_name]

    # ---------------- PREDICT BUTTON ----------------
    if st.button("Predict Penguin Species"):
        user_df = pd.DataFrame([user_input])
        pred = selected_model.predict(user_df)[0]
        proba = selected_model.predict_proba(user_df)[0]

        # Main prediction
        st.success(f"Predicted species: {pred}")

        st.markdown("---")
        st.subheader("Class Probabilities")

        # Visual class indicator: 1 = predicted class, 0 = others
        cols = st.columns(len(selected_model.classes_))

        predicted_class = selected_model.classes_[proba.argmax()]

        for col, cls in zip(cols, selected_model.classes_):
            value = 1 if cls == predicted_class else 0
            col.metric(label=cls, value=value)
