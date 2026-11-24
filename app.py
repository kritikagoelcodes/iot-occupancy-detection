import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
TARGET = "Occupancy"

st.set_page_config(
    page_title="Indoor Occupancy Detection",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 IoT-Based Indoor Occupancy Detection")
st.markdown(
    "Predict whether a room is **occupied** or **vacant** "
    "using environmental sensor readings."
)

DATA_PATH = Path(__file__).parent / "Occupancy.csv"

@st.cache_data
def load_dataset(path_string, modified_time):
    data = pd.read_csv(path_string)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))
    data = data[required].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    data[TARGET] = data[TARGET].astype(int)
    return data

@st.cache_resource
def fit_models(data):
    X = data[FEATURES]
    y = data[TARGET]

    if y.nunique() != 2:
        raise ValueError("Occupancy column must contain both classes 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    evaluation_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    evaluation_model.fit(X_train, y_train)
    test_predictions = evaluation_model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, test_predictions),
        "Precision": precision_score(y_test, test_predictions, zero_division=0),
        "Recall": recall_score(y_test, test_predictions, zero_division=0),
        "F1-score": f1_score(y_test, test_predictions, zero_division=0),
    }

    final_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    final_model.fit(X, y)
    return final_model, metrics, len(X_train), len(X_test)

if not DATA_PATH.exists():
    st.error("Occupancy.csv was not found.")
    st.info(
        "Put the file named Occupancy.csv in the same folder as app.py, "
        "then restart the app."
    )
    st.stop()

try:
    data = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    model, metrics, train_count, test_count = fit_models(data)
except Exception as error:
    st.error(f"Could not prepare the model: {error}")
    st.stop()

with st.sidebar:
    st.header("Project information")
    st.write(f"Dataset rows used: **{len(data):,}**")
    st.write(f"Training rows: **{train_count:,}**")
    st.write(f"Testing rows: **{test_count:,}**")
    st.write("Selected demo model: **Random Forest**")
    st.caption("The model is trained from Occupancy.csv when the app loads.")

st.subheader("📊 Model evaluation")
metric_cols = st.columns(4)
for col, (name, value) in zip(metric_cols, metrics.items()):
    col.metric(name, f"{value * 100:.2f}%")

st.caption(
    "Metrics are measured on a stratified held-out test set. "
    "For the interactive demo, the final Random Forest is then fitted "
    "on the full available dataset."
)

st.divider()
st.subheader("🎛️ Enter sensor readings")
st.write("Choose values within the ranges found in the dataset.")

input_cols = st.columns(2)
user_values = {}

for index, feature in enumerate(FEATURES):
    minimum = float(data[feature].min())
    maximum = float(data[feature].max())
    default = float(data[feature].median())
    column = input_cols[index % 2]

    if minimum == maximum:
        user_values[feature] = minimum
        column.number_input(
            feature, value=minimum, disabled=True, key=f"fixed_{feature}"
        )
    else:
        step = max((maximum - minimum) / 100, 0.00001)
        user_values[feature] = column.slider(
            feature,
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=step,
            key=f"sensor_{feature}"
        )

if st.button("🔍 Predict Occupancy", type="primary", use_container_width=True):
    sample = pd.DataFrame([user_values], columns=FEATURES)
    predicted_class = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    class_to_probability = dict(zip(model.classes_, probabilities))
    occupied_probability = float(class_to_probability.get(1, 0.0))

    st.divider()
    st.subheader("Prediction result")

    if predicted_class == 1:
        st.success("🟢 ROOM STATUS: OCCUPIED")
    else:
        st.warning("🔴 ROOM STATUS: VACANT")

    st.metric("Estimated probability of occupancy", f"{occupied_probability * 100:.2f}%")
    st.write("Sensor readings submitted:")
    st.dataframe(sample, use_container_width=True)

st.divider()
st.caption(
    "Demonstration mode: readings are entered manually. "
    "This application does not connect to physical sensors or provide live IoT readings."
)
