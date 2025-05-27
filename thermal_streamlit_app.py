# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:06:03 2025

@author: zzulk
"""

# thermal_streamlit_app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Thermal Simulation App", layout="centered")
st.title("🔥 Thermal Simulation & Hot Zone Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and display original image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thermal color map
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Threshold to detect hot zones
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    highlight = image.copy()
    highlight[mask == 255] = [0, 0, 255]  # Red overlay for hot zones


    # detect corners with the goodFeaturesToTrack function.
    corners = cv2.goodFeaturesToTrack(gray, 150, 0.01, 50)
    corners = corners.astype(int)

    # we iterate through each corner, 
    # making a circle at each point that we think is a corner.
    for i in corners:
        x, y = i.ravel()
        cv2.circle(gray, (x, y), 3, 255, -1)
        

        
    

        
    # Display results
    st.subheader("Simulated Thermal View")
    st.image(cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.subheader("Hot Zone Detection")
    st.image(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.subheader("corner")
    st.image(gray)
    

