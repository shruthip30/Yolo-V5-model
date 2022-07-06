
from logging import NullHandler
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt

st.title('Object Detection Model')

browse,text,camera = st.columns(3)
upload_image = browse.file_uploader('Upload the image', type = ['jpg','jpeg','png'])
text = text.subheader('           OR                ')
click_image = camera.camera_input("Take a picture")

model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'"C:\Users\shrut\Downloads\best.pt"')

image1,image2 = st.columns(2)

def display_YOLO_trained_image(img,uploaded):
    result = model(img,size=416)
    if(uploaded):
        image1.header("YOLO trained uploaded image")
        image1.image(np.squeeze(result.render()),width=300)
        image1.write(result.pandas().xyxy[0].name.tolist())
    else:   
        image2.header("YOLO trained captured image")
        image2.image(np.squeeze(result.render()),width=300)
        image2.write(result.pandas().xyxy[0].name.tolist())

if click_image:
    img = Image.open(click_image)
    display_YOLO_trained_image(img,uploaded=False)


if upload_image is not None:
    img = Image.open(upload_image)
    display_YOLO_trained_image(img,uploaded=True)
        
