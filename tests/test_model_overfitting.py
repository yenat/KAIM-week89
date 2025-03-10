import unittest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from test_data_preprocessing import TestDataPreprocessing

class TestModelOverfitting(TestDataPreprocessing):

    def test_model_overfitting(self):
        # Test if the model is not overfitting
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_fraud_train, self.y_fraud_train)
        train_acc = accuracy_score(self.y_fraud_train, model.predict(self.X_fraud_train))
        test_acc = accuracy_score(self.y_fraud_test, model.predict(self.X_fraud_test))
        self.assertAlmostEqual(train_acc, test_acc, delta=0.1)  # Allow 10% difference

if __name__ == '__main__':
    unittest.main()