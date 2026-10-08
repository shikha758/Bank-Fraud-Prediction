import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Bank Fraud Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Fraud Prediction")

st.write(
    "Enter transaction details to predict whether the transaction is fraudulent."
)


# ==========================================
# LOAD MODEL
# ==========================================

MODEL_FILE = "best_fraud_model_tuned.pkl"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):
        return None

    try:
        with open(MODEL_FILE, "rb") as file:
            obj = pickle.load(file)

        # Model itself
        if hasattr(obj, "predict"):
            return obj

        # Model inside dictionary
        if isinstance(obj, dict):

            for key in [
                "model",
                "best_model",
                "classifier",
                "fraud_model",
                "best_fraud_model",
                "estimator",
                "pipeline",
                "best_estimator"
            ]:

                if key in obj and hasattr(obj[key], "predict"):
                    return obj[key]

            # Search all values
            for value in obj.values():

                if hasattr(value, "predict"):
                    return value

        return None

    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


model = load_model()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Transaction Details")


step = st.sidebar.number_input(
    "Step",
    min_value=0,
    value=1
)


amount = st.sidebar.number_input(
    "Amount",
    min_value=0.0,
    value=1000.06
)


oldbalanceOrg = st.sidebar.number_input(
    "Old Balance Origin",
    min_value=0.0,
    value=5000.09
)


newbalanceOrig = st.sidebar.number_input(
    "New Balance Origin",
    min_value=0.0,
    value=4000.03
)


oldbalanceDest = st.sidebar.number_input(
    "Old Balance Destination",
    min_value=0.0,
    value=1000.05
)


newbalanceDest = st.sidebar.number_input(
    "New Balance Destination",
    min_value=0.0,
    value=2000.04
)


transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    [
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "DEBIT",
        "CASH_IN"
    ]
)


# ==========================================
# TRANSACTION INFORMATION
# ==========================================

st.subheader("Transaction Information")


display_data = pd.DataFrame({

    "step": [step],

    "type": [transaction_type],

    "amount": [amount],

    "oldbalanceOrg": [oldbalanceOrg],

    "newbalanceOrig": [newbalanceOrig],

    "oldbalanceDest": [oldbalanceDest],

    "newbalanceDest": [newbalanceDest]

})


st.dataframe(
    display_data,
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔍 Predict Fraud"):

    if model is None:

        st.error(
            "❌ Model could not be loaded."
        )

    else:

        try:

            # ----------------------------------
            # TYPE ENCODING
            # ----------------------------------

            type_mapping = {
                "PAYMENT": 0,
                "TRANSFER": 1,
                "CASH_OUT": 2,
                "DEBIT": 3,
                "CASH_IN": 4
            }

            type_enc = type_mapping[transaction_type]


            # ----------------------------------
            # FEATURE ENGINEERING
            # ----------------------------------

            # Log transformed amount
            log_amount = np.log1p(amount)


            # High amount flag
            is_high_amount = int(amount > 200000)


            # Hour from step
            hour = int(step % 24)


            # Night-time flag
            is_night = int(hour < 6 or hour >= 22)


            # Origin balance difference
            balance_diff_orig = (
                oldbalanceOrg - newbalanceOrig
            )


            # Destination balance difference
            balance_diff_dest = (
                newbalanceDest - oldbalanceDest
            )


            # ----------------------------------
            # CREATE MODEL INPUT
            # ----------------------------------

            input_data = pd.DataFrame({

                "step": [step],

                "amount": [amount],

                "log_amount": [log_amount],

                "is_high_amount": [is_high_amount],

                "hour": [hour],

                "is_night": [is_night],

                "balance_diff_orig": [balance_diff_orig],

                "balance_diff_dest": [balance_diff_dest],

                "type_enc": [type_enc]

            })


            # Exact feature order
            input_data = input_data[
                [
                    "step",
                    "amount",
                    "log_amount",
                    "is_high_amount",
                    "hour",
                    "is_night",
                    "balance_diff_orig",
                    "balance_diff_dest",
                    "type_enc"
                ]
            ]


            # ----------------------------------
            # SHOW FEATURES
            # ----------------------------------

            st.subheader("Model Input Features")

            st.dataframe(
                input_data,
                use_container_width=True
            )


            # ----------------------------------
            # PREDICTION
            # ----------------------------------

            prediction = model.predict(input_data)

            result = prediction[0]


            # ----------------------------------
            # RESULT
            # ----------------------------------

            if int(result) == 1:

                st.error(
                    "🚨 Fraudulent Transaction Detected!"
                )

            else:

                st.success(
                    "✅ Transaction is Not Fraudulent."
                )


            st.write(
                f"Prediction: `{result}`"
            )


        except Exception as e:

            st.error(
                f"❌ Prediction Error: {e}"
            )