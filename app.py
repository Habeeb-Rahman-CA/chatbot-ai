"""
Web interface for Basic AI Chatbot using Flask.
Run with: python3 app.py
"""

from flask import Flask, jsonify, render_template, request
from chatbot import BasicAIChatbot

app = Flask(__name__)
bot = BasicAIChatbot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    result = bot.respond(message)

    return jsonify({
        "response": result.response,
        "nlp": {
            "intent": result.intent,
            "confidence": result.confidence,
            "tokens": result.tokens,
            "extracted_entities": result.extracted_entities,
            "matched_pattern": result.matched_pattern,
            "is_fallback": result.is_fallback,
        },
        "context": {
            "user_name": bot.context.user_name,
            "turn_count": bot.context.turn_count,
        }
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    bot.reset_conversation()
    return jsonify({"status": "ok", "message": "Memory reset successfully"})


@app.route("/api/context", methods=["GET"])
def get_context():
    return jsonify({
        "user_name": bot.context.user_name,
        "turn_count": bot.context.turn_count,
        "last_intent": bot.context.last_intent,
        "history_count": len(bot.context.history)
    })


if __name__ == "__main__":
    print("Starting Chatbot Web UI on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
