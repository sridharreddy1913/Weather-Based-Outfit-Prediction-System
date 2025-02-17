from flask import Flask, render_template, request, jsonify
import joblib
import requests

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load('weather_to_outfit_model.joblib')
weather_encoder = joblib.load('weather_encoder.joblib')
outfit_encoder = joblib.load('outfit_encoder.joblib')

def get_weather_by_coords(lat, lon):
    api_key = '9595ea959414d0b095f1118c780bc0b0'  # Replace with your actual OpenWeatherMap API key
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric'
    return fetch_weather_data(url, lat, lon)

def get_weather_by_city(city):
    api_key = '9595ea959414d0b095f1118c780bc0b0'  # Replace with your actual OpenWeatherMap API key
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    return fetch_weather_data(url)

def fetch_weather_data(url, lat=None, lon=None):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather = data['weather'][0]['main']
            location = data['name']
            # Ensure lat/lon are set if using city name
            if not lat or not lon:
                lat = data['coord']['lat']
                lon = data['coord']['lon']
            return weather, location, lat, lon
        else:
            return f"Error: API request failed with status code {response.status_code}", None, None, None
    except Exception as e:
        return f"Error: An exception occurred - {e}", None, None, None

def get_model_recommendation(weather_condition):
    # Encode the weather condition
    encoded_condition = weather_encoder.transform([[weather_condition]])

    # Predict outfit components using the model
    prediction = model.predict(encoded_condition)
    
    # Decode predictions to original outfit labels
    decoded_prediction = outfit_encoder.inverse_transform(prediction)
    
    # Return predictions for Outfit1, Outfit2, and Outfit3
    return decoded_prediction[0].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    city = data.get('city')
    lat = data.get('lat')
    lon = data.get('lon')

    # Determine whether to use city or coordinates
    if city:
        weather_condition, location, lat, lon = get_weather_by_city(city)
    elif lat and lon:
        weather_condition, location, lat, lon = get_weather_by_coords(lat, lon)
    else:
        return jsonify({'error': 'Please provide a city or coordinates'}), 400
    
    # Check if any error occurred
    if weather_condition and weather_condition.startswith("Error:"):
        return jsonify({'error': weather_condition}), 400

    # Get outfit recommendations using the model
    try:
        recommendations = get_model_recommendation(weather_condition)
    except Exception as e:
        return jsonify({'error': f"An error occurred while predicting recommendations: {e}"}), 500

    # Respond with recommendations
    return jsonify({
        'weather': weather_condition,
        'location': location,
        'recommendation': recommendations,
        'lat': lat,
        'lon': lon
    })

if __name__ == '__main__':
    app.run(debug=True)
