import streamlit as st
import pickle
import numpy as np


with open('anemia_pred.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


def predict(sex, feature1, feature2, feature3, feature4):
    sex_encoded = np.array([0, 0])
    if sex == 'Male':
        sex_encoded[1] = 1
    else:
        sex_encoded[0] = 1
    sex_encoded = sex_encoded.reshape(1, -1)
    
    features = np.array([[feature1, feature2, feature3, feature4]]).astype(float)
    features_scaled = scaler.transform(features)
    
    final_features = np.hstack((sex_encoded, features_scaled))
    
    prediction = model.predict(final_features)
    return prediction[0]


male_symbol = "\u2642"
female_symbol = "\u2640"

st.title("🌟 Anemia Prediction by Classification 🌟")


st.markdown("""
<style>
.stRadio > div {flex-direction:row;}
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    padding: 10px 24px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
div.stButton > button:first-child:hover {
    background-color: #45a049;
}
.stSlider {width: 100%;}

.result-box {
    padding: 10px;
    border-radius: 5px;
    color: #fff;
    font-weight: bold;
    text-align: center;
}
.result-yes {
    background-color: #c9463c; 
}
.result-no {
    background-color: #1a691e; 
}

</style>
""", unsafe_allow_html=True)

st.markdown("### Please enter the following details:")


col1, col2 = st.columns(2)

with col1:
    sex = st.radio("Sex", [f"Male {male_symbol}", f"Female {female_symbol}"])


col1, col2 = st.columns(2)

with col1:
    feature1 = st.slider(":red[Red Pixel]", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    st.caption("Input red pixel intensity to analyze hemoglobin concentration in blood")
    feature2 = st.slider(":green[Green Pixel]", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    st.caption("Input green pixel values to detect specific features in blood samples.")

with col2:
    feature3 = st.slider(":blue[Blue Pixel]", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    st.caption("Input blue pixel data to enhance contrast for accurate feature analysis..")
    feature4 = st.slider("Hb", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    st.caption("Input measured hemoglobin level for anemia risk prediction.")

if st.button("Predict"):
    result = predict(sex.split()[0], feature1, feature2, feature3, feature4)
    if result == 1:
        st.markdown('<div class="result-box result-yes">The prediction is: Yes. There is a high risk of anemia.</div>', unsafe_allow_html=True)
    else:
         st.markdown('<div class="result-box result-no">The prediction is: No. There is a low risk of anemia.</div>', unsafe_allow_html=True)


st.markdown("""
<div class="references">
    <h3>References</h3>
    <ul>
        <li><a href="https://www.sciencedirect.com/science/article/pii/S2590093523000322#fig4" target="_blank">Reference 1</a></li>
        <li><a href="https://iopscience.iop.org/article/10.1088/1757-899X/420/1/012101/pdf" target="_blank">Reference 2</a></li>
    </ul>
</div>
""", unsafe_allow_html=True)