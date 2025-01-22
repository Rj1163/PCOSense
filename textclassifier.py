import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # Import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Load data
file_inf = "D:\\3rd Year\\Sem 6\\Minor II\\PCOS_infertility.csv"
file_woinf = "D:\\3rd Year\\Sem 6\\Minor II\\PCOS_data_without_infertility.xlsx"

PCOS_inf = pd.read_csv(file_inf)
PCOS_woinf = pd.read_excel(file_woinf, sheet_name="Full_new")

# Merge data
data = pd.merge(PCOS_woinf, PCOS_inf, on='Patient File No.', suffixes=('', '_y'), how='left')

# Drop unnecessary columns
data = data.drop(['Unnamed: 44', 'Sl. No_y', 'PCOS (Y/N)_y', '  I   beta-HCG(mIU/mL)_y',
                  'II    beta-HCG(mIU/mL)_y', 'AMH(ng/mL)_y'], axis=1)

# Replace non-numeric values with NaN in the 'II    beta-HCG(mIU/mL)' column
data['II    beta-HCG(mIU/mL)'] = pd.to_numeric(data['II    beta-HCG(mIU/mL)'], errors='coerce')

# Fill missing values with the median
data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median(), inplace=True)

# Dealing with categorical values
data["AMH(ng/mL)"] = pd.to_numeric(data["AMH(ng/mL)"], errors='coerce')

# Dealing with missing values
data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median(), inplace=True)
data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median(), inplace=True)
data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].median(), inplace=True)
data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].median(), inplace=True)

# Clearing up the extra space in the column names
data.columns = [col.strip() for col in data.columns]

# Assuming 'PCOS (Y/N)' is the target variable
target = data['PCOS (Y/N)']

# Drop non-numeric columns and target variable
numeric_features = data.select_dtypes(include=['number']).drop(columns=['Patient File No.', 'PCOS (Y/N)'])

# Fill missing values with the mean
numeric_features.fillna(numeric_features.mean(), inplace=True)

selected_features = ['Cycle(R/I)', 'Cycle length(days)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
                     'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'Age (yrs)',
                     'Weight (Kg)', 'Height(Cm)', 'BMI', 'Waist:Hip Ratio']

selected_data = numeric_features[selected_features]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_data, target, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

# Train and evaluate models
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    results[name] = scores.mean()

# Print accuracy of each model
for name, score in results.items():
    print(f"{name} Accuracy: {score:.4f} (CV Mean)")

# Choose the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"Best Model after GridSearchCV: {best_model}")

# Train the best model on the full training set
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the best model to a file
joblib.dump(best_model, "best_model.pkl")

# Model comparison graph
model_names = list(results.keys())
model_accuracies = list(results.values())

plt.figure(figsize=(10, 6))
plt.barh(model_names, model_accuracies, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Comparison')
plt.gca().invert_yaxis()  # Invert y-axis to display the highest accuracy on top
plt.show()

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']  # Adjust class weights for imbalance
}

# Instantiate the GridSearchCV object for Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the data
grid_search_rf.fit(X_train, y_train)

# Get the best parameters and best score for Random Forest
print("Best Parameters (Random Forest):", grid_search_rf.best_params_)
print("Best Score (Random Forest):", grid_search_rf.best_score_)

# Get the best Random Forest model
best_rf_classifier = grid_search_rf.best_estimator_

# Evaluate the best Random Forest model on the test set
y_pred_rf = best_rf_classifier.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Test Accuracy (Random Forest): {test_accuracy_rf:.4f}")

# Print classification report for Random Forest
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Plot confusion matrix for the best Random Forest model
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot feature importance for Random Forest
feature_importances_rf = best_rf_classifier.feature_importances_
sorted_indices_rf = np.argsort(feature_importances_rf)[::-1]  # Sort indices in descending order

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_rf[sorted_indices_rf], y=X_train.columns[sorted_indices_rf], palette='coolwarm')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for PCOS Detection (Random Forest)')
plt.show()


# Correlation matrix
plt.figure(figsize=(10, 8))
corr = numeric_features.corr()
heatmap = sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True,
                      annot_kws={"fontsize": 4})
plt.xticks(fontsize=4)  # Adjust x-axis label font size
plt.yticks(fontsize=4)
plt.title("Correlation Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

