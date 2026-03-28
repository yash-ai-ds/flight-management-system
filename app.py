import os
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

app = Flask(__name__)
# Secret key is required for displaying flash messages
app.secret_key = "flight_management_system_secret"

DATA_FILE = 'data.json'
MODEL_FILE = 'model/flight_model.pkl'

def load_data():
    """Load flight data from the JSON file."""
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_data(data):
    """Save flight data to the JSON file."""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def train_model():
    """Train the Linear Regression model to predict flight prices."""
    data = load_data()
    if len(data) < 3:
        return False # Need more data points to effectively train

    df = pd.DataFrame(data)
    # Clean data to prevent NaN crashes on older unmigrated data
    if 'departure_date' not in df.columns:
        df['departure_date'] = '2026-01-01'
    df['departure_date'] = df['departure_date'].fillna('2026-01-01')
    df['departure_time'] = df['departure_time'].fillna('00:00')
    df['source'] = df['source'].fillna('Delhi')
    df['destination'] = df['destination'].fillna('Mumbai')
    df['price'] = df['price'].fillna(0.0)
    
    # Feature engineering: extract the hour from departure_time (HH:MM format)
    df['hour'] = df['departure_time'].apply(lambda x: int(str(x).split(':')[0]))
    
    # Feature engineering: convert departure_date to day of week to find seasonal/weekend trends
    df['day_of_week'] = pd.to_datetime(df['departure_date'], errors='coerce').dt.dayofweek.fillna(0).astype(int)
    
    # Selecting relevant features
    # Input: Source, Destination, Hour, Day of Week
    X = df[['source', 'destination', 'hour', 'day_of_week']]
    # Target: Price
    y = df['price']

    # We use ColumnTransformer to one-hot encode the text features (source, destination)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['source', 'destination'])
        ],
        remainder='passthrough'
    )

    # Creating a processing and training pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train model
    model.fit(X, y)
    
    # Save the trained model
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    return True

def predict_price(source, destination, date_str, time_str):
    """Predict flight price based on the selected features using the trained model."""
    if not os.path.exists(MODEL_FILE):
        return None
        
    model = joblib.load(MODEL_FILE)
    try:
        hour = int(time_str.split(':')[0])
    except (ValueError, AttributeError, IndexError):
        hour = 0
        
    try:
        day_of_week = pd.to_datetime(date_str).dayofweek
        if pd.isna(day_of_week):
            day_of_week = 0
    except Exception:
        day_of_week = 0
    
    # Put input directly into a Pandas DataFrame format mimicking original data
    df = pd.DataFrame([{'source': source, 'destination': destination, 'hour': hour, 'day_of_week': day_of_week}])
    
    # Predict the value (It returns an array of predictions, so we select the first one)
    predicted = model.predict(df)[0]
    return max(0, round(predicted, 2))

@app.route('/')
def index():
    # Retrieve the search query parameter if it exists
    query = request.args.get('search', '')
    data = load_data()
    
    # If the user searched for something, filter the database result
    if query:
        data = [f for f in data if query.lower() in f.get('flight_number', '').lower()]

    return render_template('index.html', flights=data, search=query)

@app.route('/add', methods=['GET', 'POST'])
def add_flight():
    if request.method == 'POST':
        # Retrieve data from form
        flight_number = request.form['flight_number']
        source = request.form['source']
        destination = request.form['destination']
        departure_date = request.form['departure_date']
        departure_time = request.form['departure_time']
        price = float(request.form['price'])

        data = load_data()
        
        # Validation: Check if flight ID already exists
        if any(f.get('flight_number') == flight_number for f in data):
            flash('Flight number already exists!', 'error')
            return redirect(url_for('add_flight'))
            
        new_flight = {
            'flight_number': flight_number,
            'source': source,
            'destination': destination,
            'departure_date': departure_date,
            'departure_time': departure_time,
            'price': price
        }
        data.append(new_flight)
        save_data(data)
        
        # Retrain ML model since new data was added
        train_model() 
        
        flash('Flight added successfully!', 'success')
        return redirect(url_for('index'))
        
    return render_template('add.html')

@app.route('/update/<flight_number>', methods=['GET', 'POST'])
def update_flight(flight_number):
    data = load_data()
    flight = next((f for f in data if f.get('flight_number') == flight_number), None)
    
    if not flight:
        flash('Flight not found!', 'error')
        return redirect(url_for('index'))

    if request.method == 'POST':
        flight['source'] = request.form['source']
        flight['destination'] = request.form['destination']
        flight['departure_date'] = request.form['departure_date']
        flight['departure_time'] = request.form['departure_time']
        flight['price'] = float(request.form['price'])
        
        save_data(data)
        
        # Retrain the ML model considering the changed pricing/flight configurations
        train_model()
        
        flash('Flight updated successfully!', 'success')
        return redirect(url_for('index'))

    return render_template('update.html', flight=flight)

@app.route('/delete/<flight_number>', methods=['POST'])
def delete_flight(flight_number):
    data = load_data()
    
    # Filter out the specific flight ID to delete
    data = [f for f in data if f.get('flight_number') != flight_number]
    
    save_data(data)
    train_model() # Updating the models after removal of outliers/old data
    
    flash('Flight deleted successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        source = request.form['source']
        destination = request.form['destination']
        departure_date = request.form['departure_date']
        departure_time = request.form['departure_time']
        
        prediction = predict_price(source, destination, departure_date, departure_time)
        if prediction is None:
            flash('Model not trained yet! Add more flights to train the model.', 'error')

    return render_template('predict.html', prediction=prediction)

@app.route('/analysis')
def analysis():
    data = load_data()
    if not data:
        # Prevent errors if there's no data
        return render_template('analysis.html', avg_price=0, popular_route="N/A", total_flights=0)
        
    df = pd.DataFrame(data)
    
    # Feature 1: Get the averge meaning of flight prices across all listings 
    avg_price = round(df['price'].mean(), 2)
    
    # Feature 2: Get most frequent route using pandas grouping
    df['route'] = df['source'] + " to " + df['destination']
    popular_route = df['route'].mode()[0] if not df['route'].empty else "N/A"
    
    # Feature 3: Simply get row count
    total_flights = len(data)
    
    return render_template('analysis.html', avg_price=avg_price, popular_route=popular_route, total_flights=total_flights)

if __name__ == '__main__':
    # Force initial model training to ensure prediction works at very start
    train_model()
    # Debug = True automatically restarts server when edits are applied
    app.run(debug=True)
