import unittest
import mlflow

class TestMlflowLogging(unittest.TestCase):

    def test_mlflow_logging(self):
        # Test if MLflow logging is working correctly
        mlflow.set_experiment("Fraud Detection Models")
        with mlflow.start_run():
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            self.assertTrue(mlflow.active_run() is not None)

if __name__ == '__main__':
    unittest.main()