import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from test_data_preprocessing import TestDataPreprocessing

class TestBaselineModelPerformance(TestDataPreprocessing):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Perform a stratified train-test split
        cls.X_fraud_train, cls.X_fraud_test, cls.y_fraud_train, cls.y_fraud_test = train_test_split(
            cls.X_fraud, cls.y_fraud, test_size=0.2, random_state=42, stratify=cls.y_fraud)

    def test_class_distribution(self):
        # Check class distribution
        unique, counts_train = np.unique(self.y_fraud_train, return_counts=True)
        unique, counts_test = np.unique(self.y_fraud_test, return_counts=True)
        print(f"Train Class Distribution: {dict(zip(unique, counts_train))}")
        print(f"Test Class Distribution: {dict(zip(unique, counts_test))}")

    def test_baseline_model_performance(self):
        # Test if the model performs better than a baseline (dummy classifier)
        baseline_model = DummyClassifier(strategy="most_frequent")
        baseline_model.fit(self.X_fraud_train, self.y_fraud_train)
        baseline_preds = baseline_model.predict(self.X_fraud_test)
        baseline_acc = accuracy_score(self.y_fraud_test, baseline_preds)
        baseline_f1 = f1_score(self.y_fraud_test, baseline_preds, zero_division=1)
        
        # Using class weights to handle imbalanced data
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(self.X_fraud_train, self.y_fraud_train)
        model_preds = model.predict(self.X_fraud_test)
        model_acc = accuracy_score(self.y_fraud_test, model_preds)
        model_f1 = f1_score(self.y_fraud_test, model_preds, zero_division=1)
        model_precision = precision_score(self.y_fraud_test, model_preds, zero_division=1)
        model_recall = recall_score(self.y_fraud_test, model_preds, zero_division=1)

        print(f"Baseline Accuracy: {baseline_acc}, Baseline F1 Score: {baseline_f1}")
        print(f"Model Accuracy: {model_acc}, Model F1 Score: {model_f1}, Model Precision: {model_precision}, Model Recall: {model_recall}")

        # Check if the model performs better than baseline
        self.assertGreater(model_f1, baseline_f1)

if __name__ == '__main__':
    unittest.main()
