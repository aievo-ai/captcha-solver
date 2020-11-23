# Captcha Solver

## Introduction

This repository aims to create a simple CAPTCHA solver using OpenCV, Keras and TensorFlow. It contains the source code
and database for training. More details about the entire solution can be found at 
https://aievo.com.br/2020/11/20/deep-learning-para-quebrar-um-captcha-usando-python/ (pt/br).

## How to Run

1. Run pip3 install -r requirements.txt
2. Run python3 main.py

It will create a folder "extracted_images" containing the segmented images, and two files: captcha_model.hdf5 and 
model_labels.dat.
  