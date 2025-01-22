import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


# Load dataset
def load_data(file_path):
    data = pd.read_excel(file_path)  # Change encoding if needed
    return data


# Preprocess data
def preprocess_data(data):
    # Your preprocessing code here
    return data


# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


# Save model
def save_model(model, file_path):
    joblib.dump(model, file_path)


# Main function
def main():
    # Load dataset
    data = load_data("D:\\3rd Year\\Sem 6\\Minor II\\PCOS_data_without_infertility.xlsx")

    # Preprocess data
    processed_data = preprocess_data(data)
    X = processed_data.drop("PCOS", axis=1)
    y = processed_data["PCOS"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Save model
    save_model(model, "pcos_detection_model.pkl")


if __name__ == "__main__":
    main()
