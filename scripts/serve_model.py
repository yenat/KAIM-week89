from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# Load models

logistic_regression_model = joblib.load('../notebooks/mlruns/901575089113022840/ff9e67a75ca546cc9ed7f1c819e8f834/artifacts/Logistic Regression/model.pkl')
random_forest_model = joblib.load('../notebooks/mlruns/901575089113022840/46bc3d1e00ce4070bd98466edca7034e/artifacts/Random Forest/model.pkl')
gradient_boosting_model = joblib.load('../notebooks/mlruns/901575089113022840/e58badcec43a4e06b8ba78efc3c626d4/artifacts/Gradient Boosting/model.pkl')


app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json(force=True)

        # Convert data to numpy array
        input_data = np.array(data['input'])

        # Predict using the models
        logistic_regression_prediction = logistic_regression_model.predict_proba(input_data)[:, 1]
        random_forest_prediction = random_forest_model.predict_proba(input_data)[:, 1]
        gradient_boosting_prediction = gradient_boosting_model.predict_proba(input_data)[:, 1]

        # Log the prediction
        logger.info(f"Predictions: LR={logistic_regression_prediction}, RF={random_forest_prediction}, GB={gradient_boosting_prediction}")

        # Return the predictions
        return jsonify({
            'logistic_regression': logistic_regression_prediction.tolist(),
            'random_forest': random_forest_prediction.tolist(),
            'gradient_boosting': gradient_boosting_prediction.tolist()
        })
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
