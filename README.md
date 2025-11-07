# Model Selection Techniques in Machine Learning

This repository contains practical implementations of essential model selection and validation techniques using scikit-learn. Each notebook demonstrates a specific aspect of model validation and hyperparameter tuning for machine learning projects.

## 📚 Contents

### 1. K-Fold Cross Validation (`k-fold.ipynb`)
- **Purpose**: Introduction to K-Fold cross-validation fundamentals
- **Key Concepts**: Data splitting, train/validation indices
- **Implementation**: 5-fold cross-validation with sample data
- **Learning Outcomes**: Understanding how data is partitioned in cross-validation

### 2. Cross Validation with Pipelines (`cross_validation_k-fold.ipynb`)
- **Purpose**: Advanced cross-validation with preprocessing pipelines
- **Dataset**: California Housing dataset
- **Techniques**: 
  - Data preprocessing (imputation, scaling)
  - Pipeline creation
  - 10-fold cross-validation with R² scoring
- **Model**: Linear Regression with complete preprocessing pipeline

### 3. Grid Search Cross Validation (`grid_search_cv.ipynb`)
- **Purpose**: Hyperparameter optimization using GridSearchCV
- **Dataset**: California Housing dataset
- **Model**: Random Forest Regressor
- **Parameters Tuned**:
  - `max_depth`: [5, 10, 20]
  - `n_estimators`: [10, 50, 100]
- **Features**: 5-fold CV, parallel processing, MSE scoring

### 4. Validation and Learning Curves (`validation_and_learning_curves.ipynb`)
- **Purpose**: Model performance analysis and bias-variance tradeoff
- **Dataset**: Synthetic classification dataset (100k samples)
- **Visualizations**:
  - Validation curves (hyperparameter vs performance)
  - Learning curves (training size vs performance)
  - Scalability analysis (fit time vs training size)
  - Performance analysis (fit time vs accuracy)

## 🛠 Technologies Used

- **Python 3.13**
- **scikit-learn**: Model selection, preprocessing, algorithms
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization (validation_and_learning_curves.ipynb)

## 🎯 Learning Objectives

By working through these notebooks, you will understand:

1. **Cross-Validation Fundamentals**: How to properly split data for unbiased model evaluation
2. **Pipeline Integration**: Combining preprocessing steps with model training
3. **Hyperparameter Tuning**: Systematic approach to finding optimal model parameters
4. **Performance Analysis**: Using curves to diagnose overfitting, underfitting, and scalability issues
5. **Best Practices**: Avoiding data leakage and ensuring reproducible results

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone this repo
   cd model-selection
   ```

2. **Install dependencies**:
   ```bash
   pip install scikit-learn numpy matplotlib jupyter
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Run the notebooks** in the suggested order:
   1. `k-fold.ipynb` - Start with basics
   2. `cross_validation_k-fold.ipynb` - Learn pipeline integration
   3. `grid_search_cv.ipynb` - Master hyperparameter tuning
   4. `validation_and_learning_curves.ipynb` - Analyze model behavior

## 📊 Key Results

- **Cross-validation scores**: R² ≈ 0.602 ± 0.021 (Linear Regression on housing data)
- **Best hyperparameters**: `max_depth=20`, `n_estimators=100` (Random Forest)
- **Test performance**: MSE = 0.242 (optimized Random Forest)

## 🔍 Next Steps

- Experiment with different algorithms (SVM, Gradient Boosting)
- Try advanced techniques (Randomized Search, Bayesian Optimization)
- Implement custom scoring metrics
- Add ensemble methods and model stacking

## 📝 Notes

- All models use `random_state=43` for reproducibility
- Notebooks include detailed comments explaining each step
- Performance metrics are chosen appropriately for each task (R², MSE, Accuracy)

---
*This project is part of AI and Data Science learning curriculum focusing on practical model selection techniques.*
