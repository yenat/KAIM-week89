import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class TestModelTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load and preprocess the data
        cls.fraud_data = pd.read_csv('/home/enat/KAIM-week89/notebooks/Fraud_Data.csv')

        # Convert datetime fields
        cls.fraud_data['signup_time'] = pd.to_datetime(cls.fraud_data['signup_time'])
        cls.fraud_data['purchase_time'] = pd.to_datetime(cls.fraud_data['purchase_time'])

        # Extract datetime features
        cls.fraud_data['signup_year'] = cls.fraud_data['signup_time'].dt.year
        cls.fraud_data['signup_month'] = cls.fraud_data['signup_time'].dt.month
        cls.fraud_data['signup_day'] = cls.fraud_data['signup_time'].dt.day
        cls.fraud_data['signup_hour'] = cls.fraud_data['signup_time'].dt.hour
        cls.fraud_data['purchase_year'] = cls.fraud_data['purchase_time'].dt.year
        cls.fraud_data['purchase_month'] = cls.fraud_data['purchase_time'].dt.month
        cls.fraud_data['purchase_day'] = cls.fraud_data['purchase_time'].dt.day
        cls.fraud_data['purchase_hour'] = cls.fraud_data['purchase_time'].dt.hour

        # Drop original datetime columns
        cls.fraud_data = cls.fraud_data.drop(columns=['signup_time', 'purchase_time'])

        # Define categorical and numerical columns
        categorical_cols = ['device_id', 'source', 'browser', 'sex']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), ['purchase_value', 'age', 'signup_year', 'signup_month', 'signup_day', 'signup_hour', 'purchase_year', 'purchase_month', 'purchase_day', 'purchase_hour'])
            ],
            remainder='passthrough'
        )

        # Preprocess the data
        cls.X_fraud = preprocessor.fit_transform(cls.fraud_data.drop(columns=['class']))
        cls.y_fraud = cls.fraud_data['class']

        # Split the data into training and testing sets
        cls.X_fraud_train, cls.X_fraud_test, cls.y_fraud_train, cls.y_fraud_test = train_test_split(cls.X_fraud, cls.y_fraud, test_size=0.2, random_state=42)

    def test_model_training(self):
        # Test if the model can be trained without errors
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_fraud_train, self.y_fraud_train)
        self.assertTrue(hasattr(model, 'coef_'))

if __name__ == '__main__':
    unittest.main()