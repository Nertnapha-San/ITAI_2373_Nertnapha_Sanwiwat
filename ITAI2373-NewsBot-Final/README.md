## 🌐 Web Application

A Flask-based web interface for NewsBot is available in `Webapp/`. It lets
users paste an article and get classification, sentiment, named entities,
language detection, and a summary in one pass, plus a topic-discovery page
and a plain-English "Ask NewsBot" query interface.

**Live demo:** [add your deployed URL here once live]

**Run locally:**
```bash
cd Webapp
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:5000`. See `Webapp/README.md` for full setup,
optional model upgrades, and deployment instructions.
