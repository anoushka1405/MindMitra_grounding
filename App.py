from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from Aasha_chatbot import aasha_chatbot, get_emotion_label, build_aasha_prompt

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("🔑 Google API Key loaded is:", GOOGLE_API_KEY)

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Set up Flask
app = Flask(__name__)

# ROUTES FOR EACH PAGE
@app.route("/")
def home():
    return render_template("indexnew.html")

@app.route("/about")
def about():
    return render_template("about2.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/services")
def services():
    return render_template("services2.html")

@app.route("/privacypolicy")
def privacypolicy():
    return render_template("privacypolicy.html")

@app.route("/termsofuse")
def termsofuse():
    return render_template("termsofuse.html")

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]
    emotion = get_emotion_label(user_message)
    prompt = build_aasha_prompt(user_message, emotion)

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return jsonify({
            "reply": response.text.strip(),
            "emotion": emotion
        })
    except Exception as e:
        return jsonify({
            "reply": "Oops, I’m having trouble replying right now. Please try again later.",
            "emotion": "neutral"
        })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5050)

