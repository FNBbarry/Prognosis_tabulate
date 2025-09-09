# Prognosis_tabulate
This project implements a range of machine learning and deep learning models for the prognostic classification of patients with Cervical Spondylotic Myelopathy (CSM), based on tabular data comprising Diffusion Tensor Imaging (DTI) metrics and demographic information.

The project encompasses the training scripts implemented in Jupyter Notebook for the following models:
- Machine Learning Models​​: XGBoost, SVM, Random Forest
- ​Deep Learning Models​​: TabMLP, TabResNet, FT-Transformer

Data Preprocessing Pipeline
1. ​​Feature Engineering​​:
    - Remove irrelevant feature
    - One-hot encoding for categorical variables
    - Handle class imbalance (SMOTENC oversampling)
    - Feature standardization (StandardScaler)
4. ​​Data Splitting​​: 8:2 train-test split with stratified sampling

Model Implementation:
Machine Learning Models
- Hyperparameter optimization using Optuna
- 10-time 5-fold cross-validation
- Multiple evaluation metrics: Accuracy, Recall, Precision, F1-score, AUC
- Feature importance calculation using shap

Deep Learning Models
- Implemented using PyTorch WideDeep framework
- FT-Transformer architecture implementation
- Includes EarlyStopping callback
- Feature importance calculation using captum
- Multiple evaluation metrics: Accuracy, Recall, Precision, F1-score, AUC

Environment Setup
Install required dependencies:
```python
pip install -r requirements.txt
```
