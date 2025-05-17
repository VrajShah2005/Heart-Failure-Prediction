import re
from time import sleep
import pandas as pd
import numpy as np
import os
import base64
from io import BytesIO

import streamlit as st
from streamlit.components.v1 import html
import warnings


def run():
    st.set_page_config(
        page_title="Heart Failure Detection",
        page_icon="❤",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Function To Load Our Dataset
    @st.cache_data
    def load_model(model_path):
        return pd.read_pickle(model_path)
    
    # Function to get base64 encoded images
    def get_base64_image(image_name):
        # Create 'imgs' directory if it doesn't exist
        os.makedirs('imgs', exist_ok=True)
        
        # Define paths to image files
        image_path = f"imgs/{image_name}"
        
        # Check if image exists, if not create placeholder images
        if not os.path.exists(image_path):
            # Create placeholder images using BytesIO
            img = BytesIO()
            
            if image_name == "heart.png":
                # Create a simple heart icon (for healthy heart)
                from PIL import Image, ImageDraw
                image = Image.new('RGBA', (65, 65), (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                # Draw a green heart
                draw.polygon([(32, 15), (20, 10), (10, 20), (10, 35), (32, 55), (54, 35), (54, 20), (44, 10), (32, 15)], 
                           fill=(76, 175, 80, 255))  # Green color
                image.save(img, format='PNG')
                
            elif image_name == "hearted.png":
                # Create a simple broken heart icon
                from PIL import Image, ImageDraw
                image = Image.new('RGBA', (65, 65), (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                # Draw a red heart with a crack
                draw.polygon([(32, 15), (20, 10), (10, 20), (10, 35), (32, 55), (54, 35), (54, 20), (44, 10), (32, 15)], 
                           fill=(255, 53, 71, 255))  # Red color
                # Draw a crack in the heart
                draw.line([(32, 15), (32, 55)], fill=(255, 255, 255, 200), width=3)
                image.save(img, format='PNG')
            
            img.seek(0)
            img_bytes = img.getvalue()
            
            # Save the generated image
            with open(image_path, 'wb') as f:
                f.write(img_bytes)
                
            encoded = base64.b64encode(img_bytes).decode()
        else:
            # Load existing image file
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
                
        return f"data:image/png;base64,{encoded}"
    
    # Get base64 encoded images
    heart_healthy_img = get_base64_image("heart.png")
    heart_disease_img = get_base64_image("hearted.png")

    model = pd.read_pickle("xgboost_heart_disease_detection_v1.pkl")

    # Adding custom CSS with animations and improved visuals
    st.markdown(
        """
    <style>
        /* Base styling */
        .main {
            text-align: center;
            background-color: #0a0a0a;
            color: #f0f0f0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Credit styling */
        .creator-credit {
            font-size: 1.2rem;
            text-align: center;
            margin-top: -10px;
            margin-bottom: 20px;
            animation: colorCycle 4s infinite alternate;
            font-weight: 500;
            letter-spacing: 1px;
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
        }
        
        @keyframes colorCycle {
            0% { color: #ff3547; text-shadow: 0 0 8px rgba(255, 53, 71, 0.7); }
            33% { color: #4da6ff; text-shadow: 0 0 8px rgba(77, 166, 255, 0.7); }
            66% { color: #ff9d47; text-shadow: 0 0 8px rgba(255, 157, 71, 0.7); }
            100% { color: #ff3547; text-shadow: 0 0 8px rgba(255, 53, 71, 0.7); }
        }
        
        /* Header styling with glowing animation */
        .glowing-title {
            font-size: 3.2rem;
            font-weight: bold;
            text-align: center;
            color: #ffffff;
            text-shadow: 0 0 10px #ff3547, 0 0 20px #ff3547, 0 0 30px #ff3547;
            animation: glow 1.5s ease-in-out infinite alternate;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-top: 5px;
        }
        
        @keyframes glow {
            from {
                text-shadow: 0 0 5px #ff3547, 0 0 10px #ff3547;
            }
            to {
                text-shadow: 0 0 10px #ff3547, 0 0 20px #ff3547, 0 0 30px #ff3547;
            }
        }
        
        /* Card styling */
        .card {
            background: #111111;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 7px 25px rgba(255, 53, 71, 0.15);
        }
        
        /* Form styling */
        h3 {
            font-size: 25px;
            color: #e0e0e0;
            margin-bottom: 15px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }   
        
        .st-emotion-cache-16txtl3 h1 {
            font: bold 29px 'Segoe UI', sans-serif;
            text-align: center;
            margin-bottom: 15px;
            color: #ffffff;
        }
        
        /* Sidebar styling */
        div[data-testid=stSidebarContent] {
            background-color: #151f30;
            border-right: 4px solid #1c2840;
            padding: 8px!important;
        }

        div.block-containers {
            padding-top: 0.5rem;
        }

        .st-emotion-cache-z5fcl4 {
            padding-top: 0.5rem;
            padding-bottom: 1rem;
            padding-left: 1.1rem;
            padding-right: 2.2rem;
            overflow-x: hidden;
        }

        .st-emotion-cache-16txtl3 {
            padding: 1.5rem 0.6rem;
        }

        /* Plot container styling */
        .plot-container.plotly {
            border: 1px solid #2a3950;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .plot-container.plotly:hover {
            box-shadow: 0 0 15px rgba(255, 53, 71, 0.3);
        }

        div.st-emotion-cache-1r6slb0 span.st-emotion-cache-10trblm {
            font: bold 24px 'Segoe UI', sans-serif;
        }
        
        /* Image styling */
        div[data-testid=stImage] {
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            transition: transform 0.3s ease;
        }
        
        div[data-testid=stImage]:hover {
            transform: scale(1.05);
        }

        /* Form elements styling - improved visibility */
        div[data-baseweb=select]>div {
            cursor: pointer;
            background-color: #111111;
            border: 1px solid #333;
            border-radius: 8px;
            transition: all 0.3s ease;
            color: #ffffff;
        }
        
        div[data-baseweb=select]>div:hover {
            border-color: #ff3547;
            box-shadow: 0 0 8px rgba(255, 53, 71, 0.4);
        }

        /* Input field styling - improved visibility */
        div[data-baseweb=base-input] {
            background-color: #111111;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 5px;
            transition: all 0.3s ease;
            color: #ffffff;
        }
        
        div[data-baseweb=base-input]:focus-within {
            border-color: #ff3547;
            box-shadow: 0 0 10px rgba(255, 53, 71, 0.4);
        }
        
        /* Number input field styling */
        input[type="number"] {
            background-color: #111111 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            padding: 8px !important;
            font-size: 16px !important;
        }
        
        /* Label styling */
        label {
            color: #d0d0d0 !important;
            font-weight: 500 !important;
            font-size: 16px !important;
            margin-bottom: 8px !important;
            display: block !important;
        }

        /* Button styling with glow effect */
        div[data-testid=stFormSubmitButton] {
            display: flex;
            justify-content: center;
            width: 100%;
        }

        div[data-testid=stFormSubmitButton] > button {
            width: 40%;
            background: #111111;
            border: 2px solid #ff3547;
            padding: 18px;
            border-radius: 30px;
            font-weight: bold;
            font-size: 18px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            z-index: 1;
            color: #fff;
        }
        
        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(255, 53, 71, 0.7);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(255, 53, 71, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(255, 53, 71, 0);
            }
        }
        
        div[data-testid=stFormSubmitButton] > button:hover {
            animation: pulse 1.5s infinite;
            transform: translateY(-3px);
            box-shadow: 0 7px 15px rgba(255, 53, 71, 0.4);
        }
        
        div[data-testid=stFormSubmitButton] p {
            font-weight: bold;
            font-size: 20px;
        }
        
        .result-card {
            background: linear-gradient(145deg, #151f30, #1c2840);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
            text-align: center;
            border: 1px solid #2a3950;
            animation: fadeIn 1s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-positive {
            color: #4CAF50;
            font-size: 2.2rem;
            font-weight: bold;
            text-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
            animation: pulseText 2s infinite;
        }
        
        .result-negative {
            color: #ff3547;
            font-size: 2.2rem;
            font-weight: bold;
            text-shadow: 0 0 10px rgba(255, 53, 71, 0.5);
            animation: pulseText 2s infinite;
        }
        
        @keyframes pulseText {
            0% { opacity: 0.8; }
            50% { opacity: 1; }
            100% { opacity: 0.8; }
        }
        
        .percentage-display {
            font-size: 2.5rem;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 10px;
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.3);
        }
        
        .stAppViewBlockContainer {
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .st-emotion-cache-1v0mbdj {
            display: block;
        }
        
        .st-emotion-cache-gi0tri {
            display: none !important;
        }
        
        /* Loading animation */
        .loading-spinner {
            width: 40px;
            height: 40px;
            margin: 20px auto;
            border: 4px solid rgba(255, 53, 71, 0.3);
            border-radius: 50%;
            border-top-color: #ff3547;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Progress bar animation */
        .progress-bar {
            height: 4px;
            background: linear-gradient(90deg, #ff3547, #ff8547, #ff3547);
            background-size: 200% 100%;
            animation: gradient-shift 2s ease infinite;
            border-radius: 2px;
            margin: 10px 0;
        }
        
        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Prediction display outside box */
        .prediction-display {
            text-align: center;
            font-size: 2rem;
            margin-top: 20px;
            font-weight: bold;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }
        
        /* Heart icon styling */
        .heart-icon {
            width: 65px;
            height: 65px;
            margin: 0 auto;
            display: block;
            margin-bottom: 20px;
        }
    </style>
    """,
        unsafe_allow_html=True
    )

    header = st.container()
    content = st.container()

    with header:
        st.markdown("<h1 class='glowing-title'>Heart Failure Prediction 💔</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class="creator-credit">
            Made By : Vraj Shah
        </div>
        <div style="text-align: center; margin-bottom: 20px; animation: fadeIn 1s ease;">
            <p style="font-size: 1.2rem; color: #aaa;">
                Input your health parameters below to predict the risk of heart failure
            </p>
        </div>
        """, unsafe_allow_html=True)

    with content:
        col1, col2 = st.columns([7, 5])

        with col1:
            st.markdown("""
            <div class="card">
                <h3 style="text-align: center; margin-bottom: 25px;">Patient Information</h3>
            """, unsafe_allow_html=True)
            
            with st.form("Predict"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.number_input('Age', min_value=1,
                                          max_value=90, value=48,
                                          help="Enter patient's age (1-90)")

                    max_heart_rate = st.number_input('Max Heart Rate', min_value=0,
                                                     max_value=200, value=100,
                                                     help="Maximum heart rate achieved during exercise")

                    ecg = st.selectbox('ECG Results', options=[
                        "Normal", "ST", "LVH"], index=0,
                        help="Electrocardiogram results")

                    st_slope = st.selectbox('ST Slope', options=[
                                            "Up", "Flat", "Down"], index=0,
                                            help="Slope of the peak exercise ST segment")

                with c2:
                    blood_pressure = st.number_input('Resting Blood Pressure', min_value=0,
                                                     max_value=200, value=140,
                                                     help="Resting blood pressure in mm Hg")

                    old_peak = st.number_input('ST Depression (Old Peak)', min_value=-3.0,
                                               max_value=4.5, value=2.5,
                                               help="ST depression induced by exercise relative to rest")

                    chest_pain_type = st.selectbox('Chest Pain Type', options=[
                        "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], index=0, 
                        help="Type of chest pain experienced")

                    exercise_angina = st.selectbox(
                        'Exercise-Induced Angina', options=["No", "Yes"], index=0,
                        help="Angina induced by exercise")
                with c3:
                    cholesterol = st.number_input('Cholesterol Level', min_value=0,
                                                  max_value=510, value=228,
                                                  help="Serum cholesterol in mg/dl")

                    st.write("")

                    gender = st.selectbox('Gender', options=[
                        "Male", "Female"], index=0)

                    fasting_blood_sugar = st.selectbox('Fasting Blood Sugar', options=[
                        "Greater Than 120 mg/dl", "Less Than 120 mg/dl"], index=0,
                        help="Fasting blood sugar level")

                st.markdown("""
                <div style="display: flex; justify-content: center; margin-top: 20px;">
                """, unsafe_allow_html=True)
                
                predict_button = st.form_submit_button("Predict 🚀")
                
                st.markdown("""
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # This is the key change: Only display analysis and prediction after the button is clicked
            # We completely removed the instructional card and left this area blank until predict button is clicked
            if predict_button:
                patient_fasting_blood_sugar = 1
                if fasting_blood_sugar == "Less Than 120 mg/dl":
                    patient_fasting_blood_sugar = 0

                new_data = [age, blood_pressure, cholesterol,
                            patient_fasting_blood_sugar, max_heart_rate, old_peak]

                # Gender
                patient_gender = [1]  # Male

                if gender == "Female":
                    patient_gender = [0]  # Female

                # Chest Pain
                patient_chest_pain_type = [0, 0, 0]  # ASY

                if chest_pain_type == "Typical Angina":
                    patient_chest_pain_type = [0, 0, 1]

                elif chest_pain_type == "Atypical Angina":
                    patient_chest_pain_type = [1, 0, 0]

                elif chest_pain_type == "Non-anginal Pain":
                    patient_chest_pain_type = [0, 1, 0]

                # ECG
                patinet_ecg = [0, 0]  # LVH

                if ecg == "Normal":
                    patinet_ecg = [1, 0]

                elif ecg == "ST":
                    patinet_ecg = [0, 1]

                # ExerciseAngina
                patient_exercise_angina = [1]  # Yes

                if exercise_angina == "No":
                    patient_exercise_angina = [0]  # No

                # Slope
                patient_slope = [0, 0]  # Down
                if st_slope == "Flat":
                    patient_slope = [1, 0]
                elif st_slope == "Up":
                    patient_slope = [0, 1]

                # Appending All Data
                new_data.extend(patient_gender)
                new_data.extend(patient_chest_pain_type)
                new_data.extend(patinet_ecg)
                new_data.extend(patient_exercise_angina)
                new_data.extend(patient_slope)

                with st.spinner("Analyzing data..."):
                    predicted_value = model.predict([new_data])[0]
                    prediction_prop = np.round(
                        model.predict_proba([new_data])*100)
                    # Brief delay for visual feedback
                    sleep(0.3)

                    # Probability analysis section with centered images
                    st.markdown("""
                    <div class="card" style="margin-top: 20px;">
                        <h3 style="text-align: center; margin-bottom: 20px;">Probability Analysis</h3>
                        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                    """, unsafe_allow_html=True)
                    
                    heart_disease, no_heart_disease = st.columns(2)

                    with heart_disease:
                        # Use base64 encoded image
                        st.markdown(f"""
                            <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                                <img src="{heart_healthy_img}" class="heart-icon" alt="Healthy Heart">
                            </div>
                            <div style="text-align: center;">
                                <h4 style="color: #4CAF50; margin-bottom: 5px;">Not Heart Patient</h4>
                                <div class="percentage-display">{prediction_prop[0, 0]}%</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with no_heart_disease:
                        # Use base64 encoded image
                        st.markdown(f"""
                            <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                                <img src="{heart_disease_img}" class="heart-icon" alt="Heart Disease">
                            </div>
                            <div style="text-align: center;">
                                <h4 style="color: #ff3547; margin-bottom: 5px;">Heart Patient</h4>
                                <div class="percentage-display">{prediction_prop[0, 1]}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("""
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prediction result with capitalized He/She
                    gender_pronoun = "He" if gender == "Male" else "She"  # Capitalized first letter
                    result_status = "is a Heart Patient" if predicted_value == 1 else "is Not a Heart Patient"
                    result_symbol = "💔" if predicted_value == 1 else "💖"
                    result_color = "#ff3547" if predicted_value == 1 else "#4CAF50"
                    
                    st.markdown(f"""
                    <div class="card" style="margin-top: 20px;">
                        <h3 style="text-align: center; margin-bottom: 20px; color: #ffffff; font-size: 1.8rem;">
                            Prediction Result
                        </h3>
                    </div>
                    <div class="prediction-display" style="color: {result_color};">
                        {result_symbol} {gender_pronoun} {result_status}
                    </div>
                    """, unsafe_allow_html=True)
            # No else block - right side remains completely blank when not clicked
    
    pass


if __name__ == "__main__":
    run()