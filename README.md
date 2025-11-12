
# 🩺 Breast Cancer Text Model

A simple yet effective **Machine Learning model** built to **classify breast cancer tumors** as malignant or benign using Support Vector Machines (SVM).  
This project demonstrates a complete pipeline — from **data preprocessing and handling missing values** to **training, testing, and evaluating** a classification model.

---

## 🧠 Project Overview
Breast cancer diagnosis is a crucial medical task where early and accurate detection can save lives.  
This project aims to **analyze a breast cancer dataset** and build an **SVM-based classifier** that predicts whether a tumor is **malignant** (cancerous) or **benign** (non-cancerous), based on given cell nucleus measurements.

---

## 📂 Dataset
- **File:** `breast_cancer.csv`  
- **Source:** The dataset contains multiple features representing various cell attributes such as radius, texture, smoothness, and compactness.
- **Target Variable:**  
  - `M` → Malignant  
  - `B` → Benign  

---

## ⚙️ Technologies Used
| Category | Libraries/Tools |
|-----------|----------------|
| Data Handling | pandas, numpy |
| Visualization | matplotlib |
| Machine Learning | scikit-learn |
| Model | Support Vector Classifier (SVC) with RBF kernel |

---

## 🧩 Steps & Workflow
1. **Importing Libraries**
   - Essential Python libraries are imported for data analysis, visualization, and modeling.

2. **Data Loading**
   ```python
   df = pd.read_csv("breast_cancer.csv")
   ```

3. **Data Preprocessing**
   - Extracted independent (`X`) and dependent (`y`) variables.
   - Handled missing values using:
     ```python
     from sklearn.impute import SimpleImputer
     si = SimpleImputer(strategy='median')
     X = si.fit_transform(X)
     ```

4. **Splitting Data**
   - Divided dataset into training and testing subsets using an 80:20 ratio.

5. **Model Training**
   - Trained an **SVM classifier (RBF kernel)** on training data:
     ```python
     from sklearn.svm import SVC
     svc = SVC(kernel='rbf', degree=3)
     svc.fit(X_train, y_train)
     ```

6. **Prediction and Evaluation**
   - Predicted test data and computed:
     - Confusion Matrix  
     - Accuracy Score (both for training and entire dataset)

7. **Visualization**
   - Visualized decision boundaries using a scatter plot of the training data.

---

## 📈 Results
| Metric | Description | Result |
|---------|--------------|--------|
| `am1` | Accuracy on Test Data | ~ High Accuracy (varies by random state) |
| `am2` | Accuracy on Entire Dataset | Close to 1.0 (almost perfect classification) |

The confusion matrices show excellent separation between malignant and benign cases, indicating strong model performance.

---

## 🚀 How to Run the Project
1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/Breast-Cancer-Text-Model.git
   cd Breast-Cancer-Text-Model
   ```

2. **Install required libraries**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Python script**
   ```bash
   python breast_cancer_model.py
   ```

4. **View Outputs**
   - Accuracy metrics in console  
   - Visualization plot showing data distribution and separation

---

## 🧾 Sample Output
```
Confusion Matrix (Test Data)
[[72  1]
 [ 2 39]]
Accuracy: 0.98
```
A scatter plot displays class separation for visual understanding.

---

## 💡 Future Improvements
- Add more robust **hyperparameter tuning** (GridSearchCV).
- Use additional ML models (Random Forest, Logistic Regression) for comparison.
- Deploy using **Streamlit** or **Flask** for a user-friendly interface.

---

