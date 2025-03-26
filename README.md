# Assistant to Diagnose PCOS

This project provides a tool to assist in diagnosing Polycystic Ovary Syndrome (PCOS) using both text-based symptoms and ultrasound images. The tool leverages machine learning and image processing techniques to help predict the presence of PCOS and offers suggestions based on the analysis.

## Features
- **Data Preprocessing**: Handles both text-based and image data.
- **Machine Learning Models**: Uses a Random Forest Classifier to predict PCOS.
- **Image Processing**: Extracts features from ultrasound images using Histogram of Oriented Gradients (HOG).
- **Genetic Algorithm**: Feature selection optimized using a genetic algorithm.
- **PCOS Diagnosis**: Provides results based on both text symptoms and ultrasound images.
- **GPT Integration**: Offers explanations and suggestions based on diagnosis results.
- **User Interface**: A user-friendly Gradio interface to input data and view results.

### User Interface Example
Below is an example of the PCOS Diagnosis Tool interface:

![PCOS Diagnosis Tool Interface](./images/uygulama_ss!!.png)

This is the main interface where users can input their symptoms and upload ultrasound images for analysis.

### Diagnosis Result Example
Once the symptoms and ultrasound images are submitted, the tool provides a diagnosis and relevant suggestions:

![PCOS Diagnosis Result](./images/uygulama_ss_2.png)

The tool uses machine learning and image processing to give results and detailed explanations based on the inputs.

## Project Structure
1. **Data Preprocessing**: Cleans and processes input data, including handling missing values.
2. **Machine Learning Models**: Trains and tests a Random Forest model to classify PCOS.
3. **Image Processing**: Ultrasound images are processed using OpenCV and Scikit-Image libraries.
4. **Feature Selection**: A genetic algorithm (DEAP) optimizes feature selection for better model performance.
5. **PCOS Diagnosis**: The tool outputs whether PCOS is detected and provides related insights via GPT.
6. **Gradio Interface**: Offers a simple interface for users to input their medical symptoms and upload ultrasound images for diagnosis.
