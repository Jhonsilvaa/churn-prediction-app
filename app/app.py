"""Streamlit Application for Telecom Churn Prediction.

This module provides a web interface for predicting customer churn in a telecom company.
"""

import base64
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from pipelines.churn_pipeline import ChurnPipeline

image = Image.open(str(parent_dir / 'images' / 'churn.jpg'))
buffered = BytesIO()
image.save(buffered, format='PNG')
img_bytes = buffered.getvalue()
st.markdown(
    "<h1  style='text-align: center;'>Churn Prediction</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p  style='text-align: center;'>Application for telecom company churn prediction.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{base64.b64encode(img_bytes).decode()}' width='600'>
    </div>
    """,
    unsafe_allow_html=True,
)


parent_dir = Path(__file__).resolve().parent.parent


pipeline = ChurnPipeline(
    str(parent_dir / 'models' / 'features.joblib'),
    str(parent_dir / 'models' / 'encoder.joblib'),
    str(parent_dir / 'models' / 'scaler.joblib'),
    str(parent_dir / 'models' / 'model.joblib'),
)

with st.sidebar:

    st.markdown("### Please provide the client's information below.")
    gender = st.selectbox(
        'Gender', ['Female', 'Male'], help="Select the customer's gender."
    )
    senior_citizen = st.selectbox(
        'Senior Citizen',
        ['No', 'Yes'],
        help='Is the client 65 years or older?',
    )
    partner = st.selectbox(
        'Partner', ['No', 'Yes'], help='Does the client have a partner?'
    )
    dependents = st.selectbox(
        'Dependents',
        ['No', 'Yes'],
        help='Does the client have any dependents?',
    )
    tenure = st.number_input(
        'Tenure (months)',
        min_value=0,
        value=0,
        help='Number of months the client has stayed with the company.',
    )
    phone_service = st.selectbox(
        'Phone Service',
        ['Yes', 'No'],
        help='Is the client subscribed to a phone service?',
    )
    multiple_lines = st.selectbox(
        'Multiple Lines',
        ['No', 'Yes', 'No phone service'],
        help='Does the client have multiple phone lines?',
    )
    internet_service = st.selectbox(
        'Internet Service',
        ['DSL', 'Fiber optic', 'No'],
        help='type of internet service the client subscribes to.',
    )
    online_security = st.selectbox(
        'Online Security',
        ['No', 'Yes', 'No internet service'],
        help="Is online security included in the client's plan?",
    )
    online_backup = st.selectbox(
        'Online Backup',
        ['Yes', 'No', 'No internet service'],
        help='Does the client have online backup services?',
    )
    device_protection = st.selectbox(
        'Device Protection',
        ['No', 'Yes', 'No internet service'],
        help='Does the client have device protection?',
    )
    tech_support = st.selectbox(
        'Tech Support',
        ['No', 'Yes', 'No internet service'],
        help='Does the client have a technical support plan?',
    )
    streaming_tv = st.selectbox(
        'Streaming TV',
        ['Yes', 'No internet service', 'No'],
        help='"Is the client subscribed to a TV streaming service?',
    )
    streaming_movies = st.selectbox(
        'Streaming Movies',
        ['No', 'Yes', 'No internet service'],
        help='Is the client subscribed to a movie streaming service?',
    )
    contract = st.selectbox(
        'Contract',
        ['One year', 'Month-to-month', 'Two year'],
        help='Type of contract the client has signed.',
    )
    paperless_billing = st.selectbox(
        'Paperless Billing',
        ['No', 'Yes'],
        help='Does the client receive paperless (digital) billing?',
    )
    payment_method = st.selectbox(
        'Payment Method',
        [
            'Electronic check',
            'Credit card (automatic)',
            'Mailed check',
            'Bank transfer (automatic)',
        ],
        help="Client's preferred method of payment.",
    )
    monthly_charges = st.number_input(
        'Monthly Charges',
        min_value=0.0,
        value=0.0,
        help='Total monthly fee charged to the client.',
    )
    total_charges = st.number_input(
        'Total Charges',
        min_value=0.0,
        value=0.0,
        help='Total amount billed to the client so far.',
    )


data = {
    key: value
    for key, value in zip(
        pipeline.features,
        [
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
        ],
    )
}

st.markdown(
    """
    <style>
    div.stButton {
        display: flex;
        justify-content: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)


if st.button('Check Churn Risk'):
    prediction = pipeline.predict(data)
    if prediction['prediction'] == 1:
        st.markdown(
            f"""
    <div style='
        background-color: #f2dede;
        color: #a94442;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #a94442;
        width: 400px;
        margin: 0 auto;
        font-weight: bold;
        text-align: center;
    '>
        ⚠️ The customer is likely to cancel the service.<br>
        📉 Churn Probability: {prediction['proba']:.2%}
            </div>
        """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            f"""       
        <div style='
            background-color: #dff0d8;
            color: #3c763d;
            padding: 1rem;
            border-radius: 5px;
            border-left: 5px solid #3c763d;
            width: 400px;
            margin: 0 auto;
            font-weight: bold;
            text-align: center;'>
            ✅ The customer is unlikely to cancel the service.<br>
            📈 Retention Probability: {1 - prediction['proba']:.2%}
        </div>""",
            unsafe_allow_html=True,
        )
