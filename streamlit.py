import streamlit as st
import pickle

# ==============================
# Load vectorizer
# ==============================
VECTORIZER_PATH = r"C:\Users\devan\AI-ML\2_NLP_Project_DI_Flag\NLP_Project_DI_Flag_Code\models\vectorizer_tfidf.pkl"

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ==============================
# Load all 4 models
# ==============================
MODEL_PATHS = {
    "DI_Flag": r"C:\Users\devan\AI-ML\2_NLP_Project_DI_Flag\NLP_Project_DI_Flag_Code\models\problem statement 1 logistic_regression_model.pkl",
    "REPEAT_OBSERVATION": r"C:\Users\devan\AI-ML\2_NLP_Project_DI_Flag\NLP_Project_DI_Flag_Code\models\problem statement 2 random_forest_model.pkl",
    "Outcome": r"C:\Users\devan\AI-ML\2_NLP_Project_DI_Flag\NLP_Project_DI_Flag_Code\models\problem statement 3 logistic_regression_model.pkl",
    "ObservationRating": r"C:\Users\devan\AI-ML\2_NLP_Project_DI_Flag\NLP_Project_DI_Flag_Code\models\problem statement 4 support_vector_machine_model.pkl",
}

models = {}
for name, path in MODEL_PATHS.items():
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

# ==============================
# Label Mappings
# ==============================
label_mappings = {
    "DI_Flag": {0: "No", 1: "Yes"},
    "REPEAT_OBSERVATION": {0: "No", 1: "Yes"},
    "Outcome": {
        0: "Not Applicable",
        1: "Unsatisfactory",
        2: "Needs Improvement",
        3: "Good",
        4: "Satisfactory",
    },
    "ObservationRating": {
        0: "Recommendation",
        1: "Minor",
        2: "Major",
        3: "Critical",
    },
}

# ==============================
# Streamlit App
# ==============================
st.title("🏥 Healthcare Audit Inspection Prediction")
st.write("Paste an **auditing inspector observation** below to predict possible outcomes.")

# User input
user_input = st.text_area("Enter inspection observation:")

if st.button("Predict"):
    if user_input.strip():
        # Vectorize input
        X_input = vectorizer.transform([user_input])

        st.subheader("Predictions:")
        for column_name, model in models.items():
            try:
                pred_numeric = model.predict(X_input)[0]
                pred_label = label_mappings[column_name].get(pred_numeric, pred_numeric)

                # Show with colors/icons
                if column_name == "DI_Flag":
                    if pred_label == "Yes":
                        st.error(f"**{column_name}** → 🚨 {pred_label}")
                    else:
                        st.success(f"**{column_name}** → ✅ {pred_label}")

                elif column_name == "REPEAT_OBSERVATION":
                    if pred_label == "Yes":
                        st.warning(f"**{column_name}** → 🔄 {pred_label}")
                    else:
                        st.info(f"**{column_name}** → {pred_label}")

                elif column_name == "Outcome":
                    if pred_label == "Satisfactory":
                        st.success(f"**{column_name}** → 🟢 {pred_label}")
                    elif pred_label == "Unsatisfactory":
                        st.error(f"**{column_name}** → 🔴 {pred_label}")
                    else:
                        st.warning(f"**{column_name}** → 🟡 {pred_label}")

                elif column_name == "ObservationRating":
                    if pred_label == "Critical":
                        st.error(f"**{column_name}** → 🔥 {pred_label}")
                    elif pred_label == "Major":
                        st.warning(f"**{column_name}** → ⚠️ {pred_label}")
                    elif pred_label == "Minor":
                        st.info(f"**{column_name}** → 📝 {pred_label}")
                    else:
                        st.success(f"**{column_name}** → 💡 {pred_label}")

            except Exception as e:
                st.error(f"Error with {column_name}: {e}")
    else:
        st.warning("Please enter an observation before predicting.")
