# Chronic Kidney Disease (CKD) Analysis

## Overview

This repository presents an end-to-end analysis of Chronic Kidney Disease (CKD) using clinical data. The project covers data preprocessing, robust machine learning modeling, and insightful visualizations, aiming to support early detection and improved management of CKD.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Key Insights](#key-insights)
- [Recommendations](#recommendations)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Contributors](#contributors)

## Dataset

- **Source:** [Chronic Kidney Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
- **Records:** 400 patients
- **Attributes:** 25 clinical features (11 numerical, 14 nominal)
- **Timeframe:** ~2 months in a hospital setting

**Missing values handled via mean/mode imputation.**

## Key Insights

- Hemoglobin and RBC count are inversely correlated with CKD.
- Hypertension, albumin, and serum creatinine are strong positive indicators.
- Outliers handled with winsorization; key features selected using Recursive Feature Elimination (RFE).
- K-Nearest Neighbors achieved the best cross-validation accuracy (0.9924), outperforming Decision Tree and Random Forest.

## Recommendations

**For Healthcare Providers**
- Prioritize screening for patients with abnormal hemoglobin, serum creatinine, blood glucose, albumin, or hypertension.
- Integrate predictive models into clinical workflows for early CKD risk detection.
- Emphasize patient education on diabetes and hypertension management.

**For Patients**
- Follow medical advice and adopt preventive lifestyle changes if at risk.
- Schedule regular checkups, especially for those with known risk factors or family history.

## Project Structure

- `reports/` – Detailed PDF reports for each project phase (Part 3 report contains business analysis)
- `notebooks/` – Jupyter notebooks for data processing, modeling, and visualization
- `src/` – Python scripts for analysis and modeling

## How to Use

1. Explore interactive Jupyter notebooks in the `notebooks/` folder.
2. Review in-depth reports in `reports/`.
3. Use Python scripts in `src/` for custom analysis or to reproduce results.

## Contributors

- Donyal Emami
- Joseph Irving
- Ryan Nguyen
