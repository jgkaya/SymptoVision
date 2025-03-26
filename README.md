# PCOS Diagnosis Tool Using Machine Learning and Image Processing

This project provides a tool to assist in diagnosing Polycystic Ovary Syndrome (PCOS) using both text-based symptoms and ultrasound images. The tool leverages machine learning and image processing techniques to help predict the presence of PCOS and offers suggestions based on the analysis.

## Features
- **Data Preprocessing**: Handles both text-based and image data.
- **Machine Learning Models**: Uses a Random Forest Classifier to predict PCOS.
- **Image Processing**: Extracts features from ultrasound images using Histogram of Oriented Gradients (HOG).
- **Genetic Algorithm**: Feature selection optimized using a genetic algorithm.
- **PCOS Diagnosis**: Provides results based on both text symptoms and ultrasound images.
- **GPT Integration**: Offers explanations and suggestions based on diagnosis results.
- **User Interface**: A user-friendly Gradio interface to input data and view results.

## Project Structure
- **Data Preprocessing**: Cleans and processes input data, including handling missing values.
- **Machine Learning Models**: Trains and tests a Random Forest model to classify PCOS.
- **Image Processing**: Ultrasound images are processed using OpenCV and Scikit-Image libraries.
- **Feature Selection**: A genetic algorithm (DEAP) optimizes feature selection for better model performance.
- **PCOS Diagnosis**: The tool outputs whether PCOS is detected and provides related insights via GPT.
- **Gradio Interface**: Offers a simple interface for users to input their medical symptoms and upload ultrasound images for diagnosis.
