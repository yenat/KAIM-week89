import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import requests
import pandas as pd
import plotly.express as px

app = dash.Dash(__name__)

server = app.server

def fetch_summary():
    response = requests.get('http://localhost:5000/summary')
    return response.json()

def fetch_fraud_trends():
    response = requests.get('http://localhost:5000/fraud_trends')
    return response.json()

def fetch_fraud_geography():
    response = requests.get('http://localhost:5000/fraud_geography')
    return response.json()

def fetch_fraud_devices():
    response = requests.get('http://localhost:5000/fraud_devices')
    return response.json()

def fetch_fraud_browsers():
    response = requests.get('http://localhost:5000/fraud_browsers')
    return response.json()

app.layout = html.Div([
    html.H1('Fraud Detection Dashboard'),

    # Summary statistics
    html.Div([
        html.H3('Summary'),
        html.Div(id='summary-stats')
    ]),

    # Fraud trends over time
    html.Div([
        html.H3('Fraud Trends'),
        dcc.Graph(id='line-chart')
    ]),

    # Fraud geography
    html.Div([
        html.H3('Fraud by Geography'),
        dcc.Graph(id='geo-chart')
    ]),

    # Fraud by devices
    html.Div([
        html.H3('Fraud by Devices'),
        dcc.Graph(id='device-bar-chart')
    ]),

    # Fraud by browsers
    html.Div([
        html.H3('Fraud by Browsers'),
        dcc.Graph(id='browser-bar-chart')
    ])
])

@app.callback(
    Output('summary-stats', 'children'),
    Output('line-chart', 'figure'),
    Output('geo-chart', 'figure'),
    Output('device-bar-chart', 'figure'),
    Output('browser-bar-chart', 'figure'),
    Input('summary-stats', 'children')
)
def update_dashboard(_):
    summary = fetch_summary()
    trends = fetch_fraud_trends()
    geography = fetch_fraud_geography()
    devices = fetch_fraud_devices()
    browsers = fetch_fraud_browsers()

    trends_df = pd.DataFrame(trends)
    geography_df = pd.DataFrame(geography)
    devices_df = pd.DataFrame(devices)
    browsers_df = pd.DataFrame(browsers)

    # Summary statistics display
    summary_display = html.Div([
        html.P(f"Total Transactions: {summary['total_transactions']}"),
        html.P(f"Total Fraud Cases: {summary['total_fraud_cases']}"),
        html.P(f"Fraud Percentage: {summary['fraud_percentage']:.2f}%")
    ])

    # Line chart for fraud trends over time
    trends_fig = px.line(trends_df, x='purchase_time', y='class', title='Fraud Cases Over Time')

    # Geographic analysis chart
    geo_fig = px.scatter_geo(geography_df, locations='ip_address', size='class', title='Fraud by Geography')

    # Bar chart for fraud by devices
    device_fig = px.bar(devices_df, x='device_id', y='class', title='Fraud by Devices')

    # Bar chart for fraud by browsers
    browser_fig = px.bar(browsers_df, x='browser', y='class', title='Fraud by Browsers')

    return summary_display, trends_fig, geo_fig, device_fig, browser_fig

if __name__ == '__main__':
    app.run_server(debug=True)
