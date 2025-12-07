🏦 Credit Health Checker

![Project Banner](Final_result.png)

**Subtitle:** Predicting creditworthiness using advanced Machine Learning and Deep Learning techniques.

**OVERVIEW**

The Credit Health Checker is a data science project designed to automate and enhance the accuracy of credit score classification. By leveraging historical financial data and demographic information, this project aims to predict a customer's credit score bracket (e.g., Good, Standard, Poor).

This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology to ensure a structured approach to problem-solving, covering everything from data understanding to model deployment.

**KEY FEATURES**

* Comprehensive EDA: In-depth exploratory data analysis to understand feature distributions and correlations.
* Advanced Preprocessing: Handling missing values, outlier detection, and categorical encoding.
* Multi-Model Comparison: Implementation and benchmarking of three powerful tree-based models (XGBoost, Random Forest, LightGBM).
* Deep Learning Integration: Specific Sequential Deep Learning model to capture complex, non-linear patterns.
* High Performance: Achieved 82.0% Accuracy and 93.1% AUC-ROC with the best-performing model.

**TECH STACK**

* Language: Python 3.9+
* Data Manipulation: Pandas, NumPy
* Visualization: Matplotlib, Seaborn
* Machine Learning: Scikit-Learn, XGBoost, LightGBM
* Deep Learning: TensorFlow / Keras
* Environment: Jupyter Notebook / Google Colab

**DATASET & PREPROCESSING**

The dataset consists of financial and demographic features critical for credit assessment.

Key Preprocessing Steps:
1.  Data Cleaning: Imputed missing values and removed inconsistencies.
2.  Feature Engineering: Created new derived features to better represent financial health.
3.  Encoding: Applied One-Hot Encoding and Label Encoding for categorical variables.
4.  Scaling: Standardized numerical features for the Deep Learning model to ensure stable convergence.

**MODEL ARCHITECTURE**

We trained and evaluated four distinct models to find the optimal solution:
1.  XGBoost: Gradient boosting framework known for speed and performance.
2.  Random Forest: Ensemble learning method for robust classification.
3.  LightGBM: Efficient gradient boosting model optimized for large datasets.
4.  Sequential Deep Learning (ANN): A multi-layer neural network designed to identify latent patterns in high-dimensional data.

**PERFORMANCE EVALUATION**

We used Accuracy and AUC-ROC as our primary metrics.

![Classification Report](Classification.png)

Model Comparison:
* XGBoost: 82.0% Accuracy | 93.1% AUC-ROC (Best Model)
* Random Forest: ~80% Accuracy | High AUC-ROC
* LightGBM: ~81% Accuracy | High AUC-ROC
* Deep Learning (ANN): Experimental

**INSTALLATION & USAGE**

1.  Clone the repository.
2.  Navigate to the project directory.
3.  Install dependencies (pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, matplotlib, seaborn).
4.  Run the Notebooks:
    * For classical ML models: Open Credit_Health_Checker_final.html or the corresponding .ipynb file.
    * For the Deep Learning model: Open DL_Credit_DL_Model.ipynb.

**RESULTS**

* XGBoost proved to be the most effective model for this dataset, achieving the highest accuracy and area under the curve.
* The Deep Learning model provided interesting insights into feature interactions but required significantly more computational resources for marginal gains compared to the tree-based ensembles.
* Key determinants for credit scores were identified, helping financial institutions make more transparent decisions.

**FUTURE WORK**

* Hyperparameter Tuning: Further optimize the Deep Learning architecture using Keras Tuner.
* Deployment: Build a Streamlit or Flask API to serve the model as a web application.
* Explainability: Integrate SHAP (SHapley Additive exPlanations) values to explain individual predictions to users.
