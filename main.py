import requests
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Replace 'YOUR_API_KEY' with your actual WeatherAPI key
api_key = 'daa9bd83745c4896aa885329240111'
base_url = 'http://api.weatherapi.com/v1'

st.set_page_config(page_title="Monthly Temperature Prediction", layout="wide")
st.title("🌡️ Monthly Temperature Prediction using Combined Linear and Cubic Spline Interpolation")

def get_daily_temperatures(location, year, month):
    """Fetches daily average temperature data for a given location and month."""
    x_points = []
    y_points = []
    days_in_month = (datetime(year, month % 12 + 1, 1) - timedelta(days=1)).day  # Last day of the month

    for day in range(1, days_in_month + 1):
        date_str = f"{year}-{month:02d}-{day:02d}"
        endpoint = f'{base_url}/history.json'
        params = {
            'key': api_key,
            'q': location,
            'dt': date_str
        }
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            avg_temp = data['forecast']['forecastday'][0]['day']['avgtemp_c']
            x_points.append(day)
            y_points.append(avg_temp)
        else:
            st.error(f"Error: {response.status_code} for date {date_str}")
    
    return x_points, y_points

# Input section in the sidebar
with st.sidebar:
    st.header("📍 Location & Date Selection")
    location = st.text_input("Enter a location:", "Giridih, Jharkhand, India")
    year = st.number_input("Enter the year:", min_value=2000, max_value=2100, value=datetime.today().year)
    month = st.number_input("Enter the month:", min_value=1, max_value=12, value=datetime.today().month)
    x_pred_input = st.text_input("Enter prediction days (comma-separated):", "5.5, 15.25, 20.75")

# Parse user input into list of floats
x_pred_list = [float(x.strip()) for x in x_pred_input.split(",") if x.strip()]

# Fetch data
x_points, y_points = get_daily_temperatures(location, year, month)

# Linear Interpolation function
def linear_interpolation(x, x0, y0, x1, y1):
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)

def predict_temperature_linear(x, x_points, y_points):
    for i in range(len(x_points) - 1):
        if x_points[i] <= x <= x_points[i+1]:
            return linear_interpolation(x, x_points[i], y_points[i], x_points[i+1], y_points[i+1])
    return None

# Cubic Spline Interpolation from Scratch
def cubic_spline_coefficients(x_points, y_points):
    """Calculate cubic spline coefficients for a set of points."""
    n = len(x_points) - 1
    h = [x_points[i+1] - x_points[i] for i in range(n)]

    # Set up the system of equations
    alpha = [0] * (n + 1)
    for i in range(1, n):
        alpha[i] = (3 / h[i]) * (y_points[i+1] - y_points[i]) - (3 / h[i-1]) * (y_points[i] - y_points[i-1])

    # Coefficients of the tridiagonal matrix
    l = [1] + [0] * n
    mu = [0] * n
    z = [0] * (n + 1)

    for i in range(1, n):
        l[i] = 2 * (x_points[i+1] - x_points[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    l[n] = 1
    z[n] = 0

    # Back substitution
    b, c, d = [0] * n, [0] * (n + 1), [0] * n
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y_points[j + 1] - y_points[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    # a coefficients are the y values
    a = y_points[:-1]
    return a, b, c[:-1], d

def cubic_spline_interpolation(x, x_points, a, b, c, d):
    """Evaluate the cubic spline at x using precomputed coefficients."""
    # Find the right interval
    for i in range(len(x_points) - 1):
        if x_points[i] <= x <= x_points[i+1]:
            dx = x - x_points[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    return None

# Calculate cubic spline coefficients
a, b, c, d = cubic_spline_coefficients(x_points, y_points)

# Combined Predictions
combined_predictions = {}
for x_pred in x_pred_list:
    # Use linear interpolation within local intervals
    y_linear_pred = predict_temperature_linear(x_pred, x_points, y_points)
    # Use Cubic Spline for smooth, continuous predictions over the whole range
    y_spline_pred = cubic_spline_interpolation(x_pred, x_points, a, b, c, d)
    combined_predictions[x_pred] = (y_linear_pred, y_spline_pred)

# Display predictions
st.subheader("Prediction Results")
results = st.container()
with results:
    col1, col2, col3 = st.columns([1, 2, 2])
    for x_pred in x_pred_list:
        y_linear_pred, y_spline_pred = combined_predictions[x_pred]
        col1.markdown(f"**Day: {x_pred}**")
        col2.metric(label="Linear Prediction (°C)", value=f"{y_linear_pred:.2f}" if y_linear_pred is not None else "N/A")
        col3.metric(label="Cubic Spline Prediction (°C)", value=f"{y_spline_pred:.2f}" if y_spline_pred is not None else "N/A")

# Dense x range for smooth plot
x_dense = np.linspace(min(x_points), max(x_points), 100)
y_linear_dense = [predict_temperature_linear(x, x_points, y_points) for x in x_dense]
y_spline_dense = [cubic_spline_interpolation(x, x_points, a, b, c, d) for x in x_dense]

# Plotting
st.subheader("Interpolation Plot")
fig, ax = plt.subplots(figsize=(12, 6))
sns.set(style="whitegrid")
sns.lineplot(x=x_dense, y=y_linear_dense, linestyle='--', color='blue', label="Linear Interpolation", ax=ax)
sns.lineplot(x=x_dense, y=y_spline_dense, linestyle='-.', color='green', label="Cubic Spline Interpolation", ax=ax)
ax.plot(x_points, y_points, 'o', color='black', label="Daily Data")

# Mark predictions
for x_pred in x_pred_list:
    y_linear_pred, y_spline_pred = combined_predictions[x_pred]
    if y_linear_pred is not None:
        ax.plot(x_pred, y_linear_pred, 'ro', label=f"Linear Prediction at day {x_pred}")
    if y_spline_pred is not None:
        ax.plot(x_pred, y_spline_pred, 'bo', label=f"Cubic Spline Prediction at day {x_pred}")

# Customize plot appearance
ax.set_ylim(-10, 50)
ax.set_xlabel("Day of the Month")
ax.set_ylabel("Temperature (°C)")
ax.set_title(f"Temperature Interpolation for {location} on {year}-{month:02d}")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
st.pyplot(fig)
