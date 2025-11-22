from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
import os

app = Flask(__name__)
app.secret_key = "abc123"  # Needed for Flask session

# Initialize OpenAI client
client = OpenAI(api_key="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")

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
            {"role": "system", "content": "You are a helpful, friendly assistant."}
        ]

    # Append user message to conversation
    session["conversation"].append({"role": "user", "content": user_message})

    # Call OpenAI API with conversation history
    response = client.responses.create(
        model="gpt-4o-mini",
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





