import unittest
from test_data_preprocessing import TestDataPreprocessing
import hashlib

class TestDataLeakage(TestDataPreprocessing):

    def hash_row(self, row):
        return hashlib.md5(row.data.tobytes()).hexdigest()

    def test_data_leakage(self):
        # Hash each row in the training and testing datasets
        train_hashes = set(self.hash_row(row) for row in self.X_fraud_train)
        test_hashes = set(self.hash_row(row) for row in self.X_fraud_test)

        # Ensure no overlap between training and testing datasets
        train_test_overlap = train_hashes & test_hashes
        self.assertEqual(len(train_test_overlap), 0)

if __name__ == '__main__':
    unittest.main()
