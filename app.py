from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import google.generativeai as genai
import os 
app = Flask(__name__)
# CORS(app)
# CORS(app, resources={r"*": {"origins": "*"}})
CORS(app, supports_credentials=True, max_age=86400)
# ----------------------------
# INIT GEMINI API
# ----------------------------
# INIT GEMINI API FROM ENV
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.5-pro")

# ----------------------------
# LOAD CROP MODEL
# ----------------------------
model = pickle.load(open("crop_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ----------------------------
# CROP RECOMMENDATION
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict_crop():
    data = request.json

    features = [
        float(data["nitrogen"]),
        float(data["phosphorus"]),
        float(data["potassium"]),
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["ph"]),
        float(data["rainfall"]),
    ]

    input_array = np.array([features])
    pred = model.predict(input_array)[0]
    crop_name = encoder.inverse_transform([pred])[0]

    return jsonify({
        "recommendations": [
            f"{crop_name} - Best match based on soil and climate conditions"
        ]
    })


# ----------------------------
# CLIMATE RISK ANALYSIS (Gemini)
# ----------------------------
@app.route("/climate", methods=["POST"])
def climate_risks():
    data = request.json
    location = data["location"]

    prompt = f"""
    You are an agriculture climate expert.
    Analyze climate risks for the location: {location}.
    Identify possible:
    - Drought risk
    - Flood risk
    - Storm / Wind risk
    - Temperature extreme risk
    Provide:
    1. Risk type
    2. Severity (low / medium / high)
    3. Description
    4. Recommended actions (3-5 items)
    Return a structured JSON list.
    """

    response = model_gemini.generate_content(prompt)
    return jsonify({"risks": response.text})


# ----------------------------
# WEATHER & SOIL ANALYSIS (Gemini)
@app.route("/weather", methods=["POST"])
def weather_soil():
    data = request.json
    region = data["region"]
    soil = data["soilType"]

    prompt = f"""
    You must respond ONLY with valid JSON. 
    No explanation. No markdown. No text outside JSON.

    Generate weather & soil analysis:

    Region: {region}
    Soil: {soil}

    JSON FORMAT:
    {{
      "temperature": number,
      "humidity": number,
      "rainfall": number,
      "windSpeed": number,
      "recommendation": "string"
    }}
    """

    response = model_gemini.generate_content(prompt)

    # IMPORTANT: return RAW text, frontend will parse
    return jsonify({"weatherData": response.text})


# ----------------------------
# PESTICIDE RECOMMENDATIONS (Gemini)
# ----------------------------
@app.route("/pesticides", methods=["POST"])
def pesticides():
    data = request.json
    crop = data["cropType"]
    pest = data["pestType"]
    symptoms = data["symptoms"]

    prompt = f"""
    You are a crop protection expert. 
    Suggest pesticide recommendations for:
    Crop: {crop}
    Pest: {pest}
    Symptoms: {symptoms}
    Provide:
    - Top 2 pesticide names
    - Organic or Chemical type
    - Dosage
    - Application method
    - Safety precautions (4 items)
    Return a structured clean JSON.
    """

    response = model_gemini.generate_content(prompt)
    return jsonify({"recommendations": response.text})


if __name__ == "__main__":
    app.run(debug=True)
