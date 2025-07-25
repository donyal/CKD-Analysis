# Chronic Kidney Disease Prediction and Analysis

## Project Description

A three-part group project focused on analyzing chronic kidney disease (CKD), encompassing data preprocessing, robust modeling, and effective visualization. This repository demonstrates an end-to-end data science workflow that extracts actionable insights from clinical health data to support early CKD diagnosis and management.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Insights Summary](#insights-summary)
- [Recommendations](#recommendations)
- [Project Structure & Methodology](#project-structure--methodology)
- [Key Metrics & Segmentation](#key-metrics--segmentation)
- [How to Explore This Repository](#how-to-explore-this-repository)
- [Group Members](#group-members)

---

## Project Overview

This project investigates how clinical health data can be leveraged to improve early detection and management strategies for Chronic Kidney Disease (CKD), aiming to enhance patient outcomes and optimize healthcare resource allocation.

Through a three-part analytical process—data preparation, predictive modeling, and data visualization—we uncover the key predictors of CKD and demonstrate how machine learning can support healthcare professionals in risk stratification.

---

## Dataset Structure

The dataset was sourced from Kaggle and contains 400 patient records with 25 clinical attributes, collected over approximately two months in a hospital setting.

### Numerical Attributes (11)

- Age  
- Blood Pressure  
- Blood Glucose Random  
- Blood Urea  
- Serum Creatinine  
- Sodium  
- Potassium  
- Hemoglobin  
- Packed Cell Volume  
- White Blood Cell Count  
- Red Blood Cell Count  

### Nominal Attributes (14)

- Specific Gravity  
- Albumin  
- Sugar  
- Red Blood Cells (normal/abnormal)  
- Pus Cell (normal/abnormal)  
- Pus Cell Clumps (present/notpresent)  
- Bacteria (present/notpresent)  
- Hypertension (yes/no)  
- Diabetes Mellitus (yes/no)  
- Coronary Artery Disease (yes/no)  
- Appetite (good/poor)  
- Pedal Edema (yes/no)  
- Anemia (yes/no)  
- Classification (ckd/notckd)  

Missing values were handled during preprocessing using mean/mode imputation techniques.

---

## Insights Summary

### Critical Predictive Markers

- Hemoglobin and red blood cell (RBC) count show strong inverse relationships with CKD.
- Hypertension, albumin, and serum creatinine are strong positive indicators of CKD.
- Elevated blood urea nitrogen (BUN) and random blood glucose (RBG) levels are closely associated with impaired renal function.
- Diabetes mellitus emerged as a consistent and significant predictor of CKD progression.

### Data Quality and Robustness

- Winsorization was applied to treat outliers while retaining critical clinical significance.
- Recursive Feature Elimination (RFE) helped reduce dimensionality and identify the most impactful combination of features.

### Model Performance

- All models (Decision Tree, Random Forest, K-Nearest Neighbors) performed well.
- K-Nearest Neighbors (KNN) achieved the highest mean cross-validation accuracy (0.9924), making it the most suitable for this task.

![Model Accuracy Comparison](model-comparison.png)
---

## Recommendations

### For Healthcare Providers

- Focus CKD screening on patients with abnormal hemoglobin, serum creatinine, blood glucose, albumin, and hypertension indicators.
- Integrate predictive models into clinical systems to support early risk detection and decision-making.
- Enhance patient education on managing diabetes and hypertension.
- Use machine learning as a complement, not a replacement, to holistic clinical evaluation.

### For Patients

- Individuals at risk should follow medical advice and adopt preventive lifestyle measures.
- Routine checkups are essential for tracking CKD indicators, especially for those with known risk factors or a family history.

---

## Project Structure & Methodology

This project was conducted in three phases:

1. **Data Preprocessing**  
   - Mean/mode imputation for missing values  
   - Winsorization to address extreme outliers  
   - Recursive Feature Elimination for feature selection

2. **Data Modeling**  
   - Models: Decision Tree, Random Forest, K-Nearest Neighbors  
   - Evaluated using accuracy and cross-validation  
   - Tuned for optimal performance and generalization

3. **Data Presentation and Visualization**  
   - Feature importance, distributions, and correlations visualized for stakeholder understanding

---

## Key Metrics & Segmentation

Models were evaluated using standard classification metrics, focusing on accuracy and generalization through cross-validation. The insights support segmentation of patient populations based on CKD risk profiles, allowing for more personalized interventions.

---

## How to Explore This Repository

- `reports/` – Contains detailed PDF reports for each phase of the project. The Part 3 report offers the most comprehensive business analysis.
- `notebooks/` – Interactive Jupyter notebooks covering data processing, modeling, and visualization.
- `src/` – Standalone Python scripts used during analysis and modeling.

---

## Group Members

- Donyal Emami
- Joseph Irving  
- Ryan Nguyen

---
