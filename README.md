# ✈️ AI Flight Management System & Price Predictor (India Edition)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Active-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A full-stack, machine learning-powered **Flight Management System** built with **Python, Flask, and scikit-learn**. This project is localized for major Indian cities and uses Artificial Intelligence to predict flight costs based on historical data, domestic routing, and time/date seasonality!

---

## 🌟 Key Features

*   **🛢️ CRUD Operations**: Comprehensive **Create, Read, Update, Delete** operations for managing a live JSON-based flight database. No complex SQL installation required!
*   **🤖 AI Price Prediction**: Features a robust `scikit-learn` Linear Regression algorithm that autonomously trains on your background data to guess dynamic ticket pricing.
*   **📅 Seasonal Analytics**: Evaluates weekend vs. weekday price hiking by mathematically analyzing `departure_date`.
*   **📍 Domestic Localization**: Input parameters and routing architectures are locked to 15 major Indian airports (Delhi, Mumbai, Bangalore, Goa, etc.) to prevent typos and preserve AI precision.
*   **📊 Insights Dashboard**: Automatically computes domestic traffic insights and average INR (₹) costs utilizing `pandas` DataFrames.
*   **🛡️ Error Handling System**: Deep validation architecture that patches system `NaN` missing values, cleanses corrupted `.json` structures, and prevents 500 Server Crashes natively so you never crash.

---

## 🛠️ Tech Stack & Architecture

*   **Backend Server**: Flask (Python)
*   **Data Science / AI Algorithm**: pandas, scikit-learn, joblib
*   **Frontend**: HTML5, Vanilla CSS3, Jinja2 Templating
*   **Database**: Localized Persistent JSON Storage (`data.json`)
*   **System Architecture Pattern**: MVC (Model-View-Controller) utilizing RESTful routing.

---

## 🚀 Quickstart & Installation

Follow these simple instructions to effortlessly run the project locally on your machine.

### Prerequisites
*   Python 3.x installed
*   `pip` package manager installed

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/FlightManagementSystem.git
cd FlightManagementSystem
```

### 2. Create a Virtual Environment (Recommended but optional)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Module Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Backend Server
```bash
python app.py
```
*Wait for the initialization. Your terminal will automatically confirm the linear AI model is built and output a success state linking to the local host address:* `* Running on http://127.0.0.1:5000`

### 5. Access the Web Application
Open your web browser and execute: **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 📂 Repository Code Structure

```text
FlightManagementSystem/
│
├── app.py                 # Core Flask backend routes & logic & AI algorithm code
├── data.json              # Local persistent Database (Automatically formatted)
├── requirements.txt       # Dependencies requirement list
├── README.md              # Project documentation file (This file!)
├── .gitignore             # GitHub push exceptions
├── LICENSE                # Open-source distribution permissions
│
├── static/
│   └── style.css          # Beautiful frontend UI cascading stylesheets
└── templates/
    ├── base.html          # Core template mapping format housing navigation and alerts
    ├── index.html         # Homepage: Lists flights database 
    ├── add.html           # Input fields to inject records
    ├── update.html        # Database mutation configuration fields
    ├── predict.html       # ML Algorithm query dashboard view
    └── analysis.html      # Pandas computational statistics and UI dashboard
```

*(Note: The `model/flight_model.pkl` folder is intentionally ignored natively to save space; You don't need to push it! It dynamically regenerates itself perfectly when you boot up the server locally for the first time)*

---

## 🧠 How the Machine Learning Works (Behind the Scenes)
1.  **Ingestion & Mining:** The app automatically reads and creates a `pandas` DataFrame using your latest entries inside `data.json`.
2.  **Feature Engineering:** The software mathematically extracts the specific int value sequence from `departure_time` and dynamically scopes the `departure_date` to an integer representing the **day of the week**. This brilliantly captures the supply vs. demand price hike present on weekends.
3.  **Transformation Pipeline:** Because binary models natively do not contextualize language-based string text sequences (e.g., "Delhi"), the code applies a `ColumnTransformer` combined with a `OneHotEncoder` to categorize the city matrices into recognizable binaries.
4.  **Live Training Integrations:** The cleansed dataset runs cleanly through a **Linear Regression** Pipeline natively via Scikit. *Every single time a user adds, edits, or deletes a live flight, the system purposely throws away the outdated AI pipeline and dynamically **retrains** itself so accuracy stays perfectly attuned at 100%.*

---

## 📝 License
This project is open-source and officially available under the completely free-use [MIT License](LICENSE).
