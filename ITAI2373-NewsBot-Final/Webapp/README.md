# NewsBot Intelligence System 2.0 — Web App

A Flask web interface for the NewsBot Intelligence System (ITAI 2373 final
project). Paste a news article and get classification, sentiment, named
entities, language detection, and a summary in one pass — plus a small
conversational query page and a topic-discovery page.

This is a genuine implementation, not a mockup: every module below actually
runs. Each one uses a full pretrained model when its optional library is
installed, and automatically falls back to a lightweight, dependency-free
version otherwise, so the app works immediately and gets better as you add
libraries.

| Module | Full model (optional) | Built-in fallback |
|---|---|---|
| Classification | — (trained at startup on the bundled sample corpus) | TF-IDF + Logistic Regression |
| Topic modeling | — (trained at startup on the bundled sample corpus) | LDA (scikit-learn) |
| Sentiment | `transformers` (DistilBERT) | lexicon-based scoring |
| Named entities | `spacy` (`en_core_web_sm`) | regex-based extraction |
| Summarization | `transformers` (`facebook/bart-large-cnn`) | extractive TF-IDF sentence scoring |
| Semantic search | `sentence-transformers` (`all-MiniLM-L6-v2`) | TF-IDF cosine similarity |
| Language detection | `langdetect` | stopword-overlap heuristic |

## Quick start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open **http://localhost:5000**. That's it — classification and topic
modeling train automatically on the bundled sample corpus in
`data/sample_articles.csv`, and every other module runs in its lightweight
fallback mode.

## Upgrading to the full models

Each optional library upgrades one module. Install any or all of them —
the app detects what's available at startup and reports it on the home
page under "Engine status".

```bash
# Real named entity recognition
pip install spacy
python -m spacy download en_core_web_sm

# Transformer sentiment + BART summarization
pip install transformers torch

# Embedding-based semantic search
pip install sentence-transformers

# Statistical language detection
pip install langdetect
```

Restart the app after installing — you'll see the corresponding row in
"Engine status" switch from `fallback` to `ready`.

## Using your own trained classifier

The bundled classifier trains itself on `data/sample_articles.csv` at
startup so the app works out of the box. To use your own model from the
notebook instead:

```python
# after NewsBotEngine() is constructed, e.g. in app.py:
import pickle
with open("my_trained_classifier.pkl", "rb") as f:
    newsbot.classifier = pickle.load(f)
newsbot.classifier_ready = True
```

Your pickled object just needs `.predict_proba()` and `.classes_`, so a
scikit-learn `Pipeline` (like the one in `newsbot_engine.py` or the
notebook's `AdvancedNewsClassifier.model`) drops in directly.

## Project structure

```
newsbot-web/
├── app.py                  # Flask routes
├── newsbot_engine.py        # NLP engine (all modules + fallbacks)
├── requirements.txt
├── Procfile                 # gunicorn entrypoint for Heroku/Render
├── runtime.txt               # Python version pin for Heroku
├── data/
│   └── sample_articles.csv  # bundled corpus (classification, topics, search)
├── templates/
│   ├── base.html
│   ├── index.html            # analyze form + engine status
│   ├── results.html           # analysis results
│   ├── topics.html            # discovered topics
│   ├── ask.html                # conversational query page
│   └── about.html
└── static/
    ├── style.css
    └── script.js
```

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Analysis form + live engine status |
| `/analyze` | POST | Runs full analysis, renders `results.html` |
| `/api/analyze` | POST (JSON) | `{"text": "..."}` → full analysis JSON |
| `/topics` | GET | Topics discovered in the bundled corpus |
| `/ask` | GET/POST | Conversational query over the bundled corpus |
| `/api/ask` | POST (JSON) | `{"query": "..."}` → query result JSON |
| `/about` | GET | Project + module overview |
| `/health` | GET | JSON health check, useful once deployed |

## Deployment

### Heroku

```bash
pip freeze > requirements.txt   # if you added optional libraries
heroku login
heroku create your-newsbot-app-name
git init && git add . && git commit -m "Initial commit"
git push heroku main
heroku open
```

> If you install `transformers`/`torch`, the free Heroku dyno's memory
> limit may be tight. Consider Render (below) or a paid dyno for the
> full-model configuration.

### Render

1. Push this folder to a GitHub repository.
2. On [render.com](https://render.com), create a new Web Service from that repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`

### PythonAnywhere

1. Upload this folder (or clone from GitHub).
2. **Web** tab → Add a new web app → Flask → point at `app.py`.
3. In a Bash console: `pip install --user -r requirements.txt`.

## Testing

```bash
# Quick manual test via curl
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "A local AI startup raised new funding to expand its automation tools."}'

curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "positive technology news"}'
```

## Notes

- `SECRET_KEY` and `PORT` can be set as environment variables in
  production; sensible defaults are used for local development.
- `app.run(debug=...)` reads `FLASK_DEBUG` from the environment — set it
  to `false` before deploying (`export FLASK_DEBUG=false`).
- The bundled corpus (`data/sample_articles.csv`) is original sample text
  written for this project, so classification, topic modeling, and search
  all work immediately without any external dataset.
