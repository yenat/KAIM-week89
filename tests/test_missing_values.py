import unittest
import numpy as np
from scipy.sparse import issparse
from test_data_preprocessing import TestDataPreprocessing

class TestMissingValues(TestDataPreprocessing):

    def test_missing_values(self):
        # Ensure there are no missing values in the datasets
        if issparse(self.X_fraud):
            # For sparse matrices, check for missing values without converting to dense
            self.assertFalse(np.isnan(self.X_fraud.data).any())  # Check only non-zero elements
        else:
            # For dense arrays, check for missing values directly
            self.assertFalse(np.isnan(self.X_fraud).any())

if __name__ == '__main__':
    unittest.main()