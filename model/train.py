from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

def train_model(data_path: str, model_path: str):
    """
    Train a text classification model using SVM and save the trained model.
    
    Parameters:
        data_path (str): Path to the dataset CSV file.
        model_path (str): Path to save the trained model.
    """
    try:
        # Load dataset
        data = pd.read_csv(data_path)
        if 'text' not in data.columns or 'label' not in data.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        X, y = data['text'], data['label']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svm', SVC(kernel='linear', probability=True))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Test model
        y_pred = pipeline.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        
        # Save model
        joblib.dump(pipeline, model_path)
        print(f"Model saved at {model_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Specify paths
    dataset_path = r"C:\Users\PC\Desktop\Jio Institute\Quater 3\ml_engineer\Assignment1\text_classification\Text_Classification_Dataset.csv"
    model_save_path = r"C:\Users\PC\Desktop\Jio Institute\Quater 3\ml_engineer\Assignment1\text_classification\model\svm_model.pkl"
    
    # Train the model
    train_model(dataset_path, model_save_path)
