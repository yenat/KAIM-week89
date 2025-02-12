# E-Commerce and Banking Fraud Detection

This project involves:

- **Analyzing and Preprocessing Transaction Data**:
  - Handling missing values
  - Data cleaning
  - Removing duplicates
  - Correcting data types
  - Exploratory data analysis (EDA)
  
- **Creating and Engineering Features that Help Identify Fraud Patterns**:
  - Transaction frequency and velocity
  - Time-based features such as hour of day and day of week
  - Geolocation analysis by mapping IP addresses to countries
  
- **Building and Training Machine Learning Models to Detect Fraud**:
  - Selecting appropriate algorithms
  - Training and tuning models
  - Handling imbalanced data
  
- **Evaluating Model Performance and Making Necessary Improvements**:
  - Performance metrics (accuracy, precision, recall, F1-score)
  - Cross-validation and model validation techniques
  
- **Deploying the Models for Real-Time Fraud Detection and Setting Up Monitoring for Continuous Improvement**:
  - Real-time deployment of fraud detection models
  - Setting up monitoring and alert systems for continuous performance tracking and improvement

## Installation

To install the necessary dependencies, use the following command:
```bash
pip install -r requirements.txt
git clone https://github.com/yenat/KAIM-week89.git

cd KAIM-week89
jupyter notebook notebooks/task1.ipynb
jupyter notebook notebooks/task2.ipynb
jupyter notebook notebooks/task3.ipynb
```

## Running Flask Backend

To start the Flask backend, run the following command:
```bash
python app.py
```

## Running Dash Frontend

To start the Dash frontend, open a separate terminal and run the following command:
```bash
python dashboard.py
```

## Docker Setup

To set up the Docker container for this project, use the following commands:
```bash
# Build the Docker image
docker build -t fraud_detection .

# Run the Docker container
docker run -p 5000:5000 fraud_detection
```

## API Endpoints

The Flask backend provides the following API endpoints:

- `/summary`: Returns summary statistics of the fraud data
- `/fraud_trends`: Returns fraud trends over time
- `/fraud_geography`: Returns geographic analysis of fraud cases
- `/fraud_devices`: Returns fraud cases by devices
- `/fraud_browsers`: Returns fraud cases by browsers

## Conclusion

This project aims to provide a comprehensive solution for detecting and analyzing fraud in e-commerce and banking transactions. By leveraging Flask for the backend, Dash for visualizations, and Jupyter notebooks for exploratory data analysis and model building, we can achieve real-time fraud detection and continuous performance monitoring.
```
