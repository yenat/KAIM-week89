from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

def load_data():
    df = pd.read_csv('../notebooks/Fraud_Data.csv')
    return df

@app.route('/summary')
def summary():
    df = load_data()
    total_transactions = len(df)
    total_fraud_cases = df[df['class'] == 1].shape[0]
    fraud_percentage = (total_fraud_cases / total_transactions) * 100

    summary_stats = {
        'total_transactions': total_transactions,
        'total_fraud_cases': total_fraud_cases,
        'fraud_percentage': fraud_percentage
    }

    return jsonify(summary_stats)

@app.route('/fraud_trends')
def fraud_trends():
    df = load_data()
    trends = df.groupby('purchase_time')['class'].sum().reset_index()
    return jsonify(trends.to_dict(orient='records'))

@app.route('/fraud_geography')
def fraud_geography():
    df = load_data()
    geography = df.groupby('ip_address')['class'].sum().reset_index()
    return jsonify(geography.to_dict(orient='records'))

@app.route('/fraud_devices')
def fraud_devices():
    df = load_data()
    devices = df.groupby('device_id')['class'].sum().reset_index()
    return jsonify(devices.to_dict(orient='records'))

@app.route('/fraud_browsers')
def fraud_browsers():
    df = load_data()
    browsers = df.groupby('browser')['class'].sum().reset_index()
    return jsonify(browsers.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
