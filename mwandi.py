import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 1. Fetch Match Data
def fetch_match_data(url):
    try:
        logging.info("Fetching data from the website...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Example structure: Adjust this based on the website
        matches = soup.find_all('div', class_='match-container')
        data = []
        for match in matches:
            try:
                home_team = match.find('span', class_='home-team').get_text(strip=True)
                away_team = match.find('span', class_='away-team').get_text(strip=True)
                home_score = int(match.find('span', class_='home-score').get_text(strip=True))
                away_score = int(match.find('span', class_='away-score').get_text(strip=True))
                date = match.find('span', class_='match-date').get_text(strip=True)
                result = (
                    "Win" if home_score > away_score else
                    "Loss" if home_score < away_score else
                    "Draw"
                )
                data.append([home_team, away_team, home_score, away_score, date, result])
            except AttributeError:
                continue
        logging.info("Data fetched successfully!")
        return pd.DataFrame(data, columns=['Home Team', 'Away Team', 'Home Goals', 'Away Goals', 'Date', 'Result'])
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# 2. Preprocess Data
def preprocess_data(df):
    logging.info("Preprocessing data...")
    # Encode target variable
    df['Result'] = df['Result'].map({'Win': 2, 'Draw': 1, 'Loss': 0})
    
    # Add example features (add your own based on available data)
    df['Home Win Rate'] = np.random.rand(len(df))  # Simulated example
    df['Away Win Rate'] = np.random.rand(len(df))  # Simulated example
    df['Home Form'] = np.random.rand(len(df))      # Simulated example
    df['Away Form'] = np.random.rand(len(df))      # Simulated example
    df['Home Advantage'] = 1  # Binary indicator for home games

    # Features and target
    X = df[['Home Win Rate', 'Away Win Rate', 'Home Form', 'Away Form', 'Home Advantage']]
    y = df['Result']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logging.info("Preprocessing complete.")
    return X_scaled, y, scaler

# 3. Train Model
def train_model(X, y):
    logging.info("Training the model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    return model

# 4. Save Model
def save_model(model, scaler, model_file='football_model.pkl', scaler_file='scaler.pkl'):
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    logging.info(f"Model saved as {model_file}.")
    logging.info(f"Scaler saved as {scaler_file}.")

# 5. Predict Matches
def predict_match(model_file, scaler_file, match_data):
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    scaled_data = scaler.transform(match_data)
    prediction = model.predict(scaled_data)
    outcomes = {2: "Win", 1: "Draw", 0: "Loss"}
    return [outcomes[p] for p in prediction]

# Main Script
if __name__ == "__main__":
    # Fetch data
    url = 'https://example.com'  # Replace with the actual URL
    df = fetch_match_data(url)
    
    # Ensure we have data
    if df.empty:
        logging.error("No data to process. Exiting.")
    else:
        # Preprocess
        X, y, scaler = preprocess_data(df)

        # Train
        model = train_model(X, y)

        # Save model
        save_model(model, scaler)

        # Predict (example match data)
        future_matches = pd.DataFrame([{
            'Home Win Rate': 0.65,
            'Away Win Rate': 0.45,
            'Home Form': 0.7,
            'Away Form': 0.6,
            'Home Advantage': 1,
        }])
        predictions = predict_match('football_model.pkl', 'scaler.pkl', future_matches)
        print("Predicted Outcomes:", predictions)
