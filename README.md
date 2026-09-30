# ASL alphabet recognition

opencv-python==4.7.0.68
mediapipe==0.9.0.1
scikit-learn==1.2.0


Python computer vision project for recognising individual alphabet hand signs from a live webcam feed

## Overview

This project explores a lightweight approach to gesture recognition. Instead of classifying raw images, we use MediaPipe to extract hand landmarks and train a Random Forest to associate hand configurations. The trained model then displays its predictions in real time using OpenCV. 

## How it works

1. collect_imgs.py: take 200 labelled webcam images per letter (up to 5,200 img), organized in data/A/ through data/Z/
2. create_dataset.py: MediaPipe detects 21 hand landmarks per valid image. x and y are translated relative to the hand's min x and y, reducing dependence on its location in the frame. The resulting 42-feature vectors and labels are saved to data.pickle and images without a detected hand are skipped
3. train_classifier.py: train a scikit-learn RandomForestClassifier with a 80/20 split, report test accuracy, and save the model to model.p
4. inference_classifier.py: apply the landmark preprocessing to webcam frames, predict a letter, and display it beside the detected hand

## Potential applications

- accessible input methods
- educational tools for practicing hand signs
- touch-free gesture controls

## What I learnt

- practical experience with a supervised ML workflow
- how an appropriate representation—hand geometry instead of pixels can simplify a computer vision problem
- importance of data quality and testing beyond the original collection conditions

## Limitations
- classifier handles individual frames rather than motion, so no representation of dynamic signs
- training assumes one hand but current inference loop can combine landmarks from multiple hands --> input-size mismatch
- needs temporal prediction smoothing
