# -*- coding: utf-8 -*-

from logging import NullHandler
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt

st.title('Object Detection Model')

uploaded_image = st.file_uploader('Upload the image you want to run through the ML Object Detection model', type = ['jpg','jpeg','png'],accept_multiple_files=True)

cols = st.columns(2)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'"C:\Users\shrut\Downloads\best.pt"')

count = 0
for uploaded_file in uploaded_image:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # st.image(image, caption='Fashion Product Image',width=200)
        result = model(image,size=640)
        cols[count].header("YOLO trained image")
        cols[count].subheader(count+1)
        cols[count].image(np.squeeze(result.render()),width=300)
        count = count + 1
        
