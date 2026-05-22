# app.py - Streamlit Deployment UI for Client 27
import streamlit as st
import os
from PIL import Image
from predict import predict_single_image

st.set_page_config(page_title="Height Detector Portal 27", page_icon="🍀")
st.title("🍀 Space Tracker Console — Version 27")
st.sidebar.info("Backbone: NASNetMobile\nTheme Style: Emerald")

file_upload = st.file_uploader("Upload target image:", type=['png', 'jpg', 'jpeg'])
if file_upload is not None:
    img_render = Image.open(file_upload)
    st.image(img_render, use_container_width=True)
    if st.button("Run Evaluation Metrics"):
        temp_file = "temp_inference_data.jpg"
        img_render.convert("RGB").save(temp_file)
        try:
            class_result, confidence = predict_single_image(temp_file)
            st.success(f"Prediction Output Matrix: **{class_result}** ({confidence:.1f}%)")
        except Exception as e:
            st.error(f"Execution Error: {str(e)}. Make sure you run train.py first.")
        finally:
            if os.path.exists(temp_file): os.remove(temp_file)
