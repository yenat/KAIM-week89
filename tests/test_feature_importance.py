import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from test_data_preprocessing import TestDataPreprocessing

class TestFeatureImportance(TestDataPreprocessing):

    def test_feature_importance(self):
        # Test if the model assigns non-zero importance to at least some features
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_fraud_train, self.y_fraud_train)
        self.assertTrue(np.any(model.coef_ != 0))  # At least one feature has non-zero importance

if __name__ == '__main__':
    unittest.main()