from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
app.secret_key = "abc123"  # Needed for Flask session
CORS(app, supports_credentials=True)

# Initialize OpenAI client
client = OpenAI(api_key="APIKEYHERE")

# Serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Initialize conversation in session if not already
    if "conversation" not in session:
        session["conversation"] = [
            {"role": "system", "content": "You are a helpful, friendly medical assistant, designed to help with posture and exercises that help with everyday health."}
        ]

    # Append user message to conversation
    session["conversation"].append({"role": "user", "content": user_message})


    instructions = ("You are a helpful, friendly medical assistant, designed to help with posture and exercises that help with everyday health, print only in point form, under 40 words.")
    # Call OpenAI API with conversation history
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=session["conversation"]
    )

    bot_reply = response.output_text

    # Append bot reply to conversation
    session["conversation"].append({"role": "assistant", "content": bot_reply})

    return jsonify({"reply": bot_reply})

# Optional: clear conversation endpoint
@app.route("/clear", methods=["POST"])
def clear():
    session.pop("conversation", None)
    return jsonify({"status": "Conversation cleared."})

if __name__ == "__main__":
    app.run(debug=True)


