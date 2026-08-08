"""
app.py - NewsBot Intelligence System web interface.

Run locally:
    pip install -r requirements.txt
    python app.py
    -> open http://localhost:5000

See README.md for optional heavy-dependency installs (spaCy, transformers,
sentence-transformers, langdetect) that upgrade each module from its
lightweight fallback to the full model used in the notebook.
"""

import os
from flask import Flask, render_template, request, jsonify, flash

from newsbot_engine import NewsBotEngine

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

print("Initializing NewsBot engine (this may take a moment)...")
newsbot = NewsBotEngine()
print("NewsBot engine ready.")
for component, state in newsbot.status.items():
    print(f"  - {component}: {state}")


@app.route("/")
def home():
    return render_template("index.html", status=newsbot.status)


@app.route("/analyze", methods=["POST"])
def analyze_text():
    text = request.form.get("text", "").strip()

    if not text:
        flash("Please enter some text to analyze.", "error")
        return render_template("index.html", status=newsbot.status)

    result = newsbot.analyze_complete(text)

    if "error" in result:
        flash(result["error"], "error")
        return render_template("index.html", status=newsbot.status)

    return render_template("results.html", result=result)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """JSON API for programmatic access. POST {"text": "..."}"""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = newsbot.analyze_complete(text)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/topics")
def topics():
    """Module A: shows topics discovered from the bundled sample corpus."""
    discovered = newsbot.get_topics()
    return render_template("topics.html", topics=discovered, status=newsbot.status)


@app.route("/ask", methods=["GET", "POST"])
def ask():
    """Module D: simple conversational query interface over the bundled corpus."""
    answer = None
    query = ""
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            answer = newsbot.answer_query(query)
        else:
            flash("Please enter a question, e.g. \"positive technology news\".", "error")
    return render_template("ask.html", answer=answer, query=query)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    return jsonify(newsbot.answer_query(query))


@app.route("/about")
def about():
    return render_template("about.html", status=newsbot.status)


@app.route("/health")
def health():
    """Simple health check endpoint, useful once deployed."""
    return jsonify({"status": "ok", "components": newsbot.status})


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
