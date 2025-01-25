# Loan Default Prediction Using Neural Networks

## Overview

This project focuses on predicting whether a borrower will repay their loan or default, using a neural network model built with Keras. The dataset is a subset of LendingClub's loan data, which includes features such as loan amount, interest rate, borrower credit score, and more. The final model achieved 92% accuracy, offering a reliable solution for loan default risk assessment.

## Dataset

- Source: LendingClub Dataset (Kaggle) "with some extra feature engineering to do"
  
- number of rows: +300,000 row

- Features: Includes borrower-related data such as:

  - loan_amnt: Loan amount requested.

  - int_rate: Interest rate on the loan.
  
  - annual_inc: Borrower’s annual income.
  
  - dti: Debt-to-income ratio.
  
  - loan_status: Target variable indicating repayment status.

## Project Pipeline

**1. Exploratory Data Analysis (EDA)**

- Explored correlations between loan features and repayment status.

- Visualized distributions, trends, and outliers using Matplotlib and Seaborn.

**2. Data Preprocessing**

- Handled missing values and outliers.

- Encoded categorical variables (e.g., home_ownership, verification_status).

- Normalized numerical features for improved model training.

- Addressed class imbalance using **SMOTE** to ensure equitable representation of default cases.

**3. Neural Network Model**

- Architecture:

  - Input layer for numerical and categorical features.
  
  - Two hidden layers with ReLU activation.
  
  - Dropout layers to prevent overfitting.
  
  - Output layer with a sigmoid activation for binary classification.

- Hyperparameters:

  - Optimizer: Adam
  
  - Loss function: Binary Crossentropy
  
  - Metrics: Accuracy, Precision, Recall, F1 Score

**4. Model Training and Evaluation**

- Split the dataset into training and testing sets (80/20 split).

- Trained the model for 100 epochs with a batch size of 256.

Evaluated performance using metrics such as accuracy, precision, recall, and F1-score.

## Results

- Achieved 92% accuracy on the test set.

- Improved prediction of loan defaults with balanced precision and recall.

## Technologies Used

- Languages: Python

- Libraries: Pandas, NumPy, Matplotlib, Seaborn, TensorFlow/Keras, SMOTE (imbalanced-learn)
