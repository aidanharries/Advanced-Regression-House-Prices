# House Prices: Advanced Regression Techniques Using PyTorch

## Project Overview

**Objective:** Predict house prices using advanced regression techniques.

**Dataset:** Ames Housing dataset with 79 explanatory variables detailing various aspects of residential homes in Ames, Iowa.

**Acknowledgement:** Dataset compiled by Dean De Cock as an educational resource.

## Environment Setup

- **Workspace Configuration:** Utilized Google Drive for data access and storage in a Google Colab environment.

## Library Imports

- **Data Handling and Visualization:** `numpy`, `pandas`, `missingno`, `seaborn`, `matplotlib`
- **PyTorch:** `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data`
- **Sklearn:** `train_test_split`, `KFold`

## Data Preprocessing

1. **Loading Dataset:** Loaded the training dataset from Google Drive.
2. **Initial Observation:** Observed the dataset dimensions and first few rows.
3. **Exploratory Data Analysis (EDA):**
   - Visualized the distribution of the target variable 'SalePrice'.
   - Visualized missing values in the dataset.
4. **Data Cleaning and Transformation:**
   - Dropped irrelevant columns.
   - Converted categorical variables into dummy variables.
   - Normalized numerical features.
   - Handled missing values.
5. **Data Splitting:** Split the dataset into training and validation sets (80% training, 20% validation).

## Model Building

### Neural Network Architecture

Defined a custom neural network using PyTorch with the following layers:
- Input Layer
- Multiple Hidden Layers with Softplus Activation
- Output Layer

### Training Process

1. **Environment Initialization:** Set random seeds for reproducibility and determined the computing device (GPU/CPU).
2. **Data Preparation:** Converted data to PyTorch tensors and created DataLoader for batch processing.
3. **Model Initialization:** Defined the model, optimizer (SGD), and loss function (MSELoss).
4. **Training Loop:** Trained the model for 250 epochs, tracking training and validation losses.

## Model Evaluation

- **Loss Visualization:** Visualized training and validation loss over epochs.
- **Error Metrics:** Calculated Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for training and validation sets.

## Test Data Predictions

1. **Loading & Preprocessing:** Loaded and preprocessed the test dataset.
2. **Handling Column Differences:** Ensured consistent input dimensions between training and test sets.
3. **Prediction:** Used the trained model to predict house prices on the test dataset.
4. **Results Compilation:** Compiled predictions into a DataFrame.

## Conclusion

This project utilized the Ames Housing dataset to build an advanced regression model using PyTorch. The model demonstrated promising accuracy levels in predicting house prices, with reasonably low MAE and RMSE values for both training and validation sets. Future work could explore feature importance analysis, model variations, and advanced architectures to further enhance prediction accuracy.
