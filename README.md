# **Image Processing with Python: Applications in Machine Learning**

## **Overview**

This project demonstrates how to preprocess and prepare image datasets for use in machine learning algorithms, particularly focusing on extracting quantifiable features from images using image processing techniques. The goal is to develop a classifier that can effectively distinguish between different types of plant leaves by using machine learning, while also ensuring that the model is interpretable.

## **Project Objective**

We aim to apply image processing techniques to prepare data, extract features, and train a machine learning model to classify dried plant leaves into three categories:
- **PlantA**
- **PlantB**
- **PlantC**

## **Technologies Used**

- **Python**: The main programming language used to implement the image processing and machine learning pipeline.
- **Libraries**:
  - `numpy`, `pandas`, `matplotlib`: For data manipulation, visualization, and plotting.
  - `skimage`: For image processing tasks like thresholding, morphological operations, and feature extraction.
  - `scikit-learn`: For model training and splitting the dataset.
  - `RandomForestClassifier`: A machine learning model used for classification.

## **Steps to Run the Project**

### 1. **Install Dependencies**
Before running the code, make sure you have the required Python libraries installed. You can install them using `pip`:

```bash
pip install numpy pandas matplotlib scikit-image scikit-learn
```

### 2. **Dataset Preparation**
You will need a dataset of plant leaf images. The images should be in `.jpg` format, and organized in subfolders representing different plant types (e.g., `plantA`, `plantB`, `plantC`).

### 3. **Run the Code**
Once the dataset is ready, you can run the code following the steps outlined in the notebook to:
- Perform **Exploratory Data Analysis (EDA)** on the dataset.
- Apply **Image Binarization** using Otsu's method to segment the leaf from the background.
- Perform **Morphological Operations** (e.g., area closing and opening) to clean the image.
- **Label Regions** of interest using connected components.
- **Extract Features** from each region using properties like area, perimeter, and intensity.
- **Train a Random Forest Classifier** using the extracted features.

### 4. **View Results**
After training, the model's performance will be evaluated using accuracy on a test set. The most important features contributing to the classification will also be displayed.

## **File Structure**

```
/dataset
    ├── plantA/
    ├── plantB/
    └── plantC/
  
/code
    └── image_processing_ml.py
README.md
```

- `dataset/`: Folder containing the plant leaf images divided into categories (`plantA`, `plantB`, `plantC`).
- `code/`: Folder containing the main Python script (`image_processing_ml.py`) which includes the image processing steps and machine learning pipeline.

## **Main Features and Functions**

### **1. Exploratory Data Analysis (EDA)**  
The first step is to load and visualize the images to understand the dataset and identify key characteristics, such as the shape of the leaves.

### **2. Image Binarization**  
The images are converted to grayscale and thresholded using Otsu's method to separate the leaf from the background.

### **3. Morphological Operations**  
The image is cleaned using area-closing and area-opening operations to remove noise and fill holes in the leaf regions.

### **4. Region Labeling**  
Connected component labeling is applied to identify distinct regions in the image, with each region corresponding to a part of the leaf.

### **5. Feature Extraction**  
Quantifiable properties such as area, convex area, eccentricity, and perimeter are extracted for each region to use as features for the machine learning model.

### **6. Feature Engineering**  
Additional features are derived from the extracted properties, such as ratios between areas and perimeter, to enhance the classifier's ability to distinguish between leaf types.

### **7. Model Training**  
A Random Forest Classifier is trained using the extracted and engineered features to classify the leaf images into their respective plant types.

### **8. Model Evaluation**  
The model's accuracy on the test dataset is calculated, and the most important features contributing to the classification are identified.

## **Results**

The Random Forest Classifier achieved a test accuracy of **90%**. The key features for classification were identified as:
- **mean_intensity**
- **area_ratio_convex**
- **solidity**
- **perimeter_ratio_major**
- **peri_over_dia**

These features reflect characteristics such as the leaf's texture, shape, and edge ruggedness, which were identified during the Exploratory Data Analysis (EDA).

## **Conclusion**

This project successfully demonstrates the power of combining image processing techniques with machine learning to develop an interpretable classifier for plant leaf classification. The Random Forest model performed well, and the feature extraction process helped in understanding the importance of specific leaf characteristics for classification.

## **Future Work**

- Explore additional machine learning algorithms such as SVM or Neural Networks to improve the classification accuracy.
- Incorporate more advanced image processing techniques such as edge detection and contour analysis for more robust feature extraction.
- Experiment with data augmentation to enhance the training set, especially for cases with limited data.
