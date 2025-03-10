import unittest
from test_data_preprocessing import TestDataPreprocessing

class TestClassImbalance(TestDataPreprocessing):

    @classmethod
    def setUpClass(cls):
        # Call the parent class's setUpClass method to initialize the data
        super().setUpClass()
        print("Debug: y_fraud in TestClassImbalance:", cls.y_fraud)  # Debugging statement

    def test_class_imbalance(self):
        # Check if the target variable is not highly imbalanced
        print("Debug: y_fraud in test_class_imbalance:", self.y_fraud)  # Debugging statement
        fraud_class_ratio = self.y_fraud.mean()
        self.assertGreater(fraud_class_ratio, 0.01)  # At least 1% positive class

if __name__ == '__main__':
    unittest.main()
