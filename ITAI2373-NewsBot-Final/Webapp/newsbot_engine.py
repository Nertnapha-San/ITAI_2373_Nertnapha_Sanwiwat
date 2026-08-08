"""
newsbot_engine.py

Core NLP engine for the NewsBot Intelligence System web app.

This wraps the same techniques used in the NewsBot 2.0 notebook
(classification, topic modeling, sentiment, entity extraction,
summarization, semantic search, language detection, and a simple
conversational query layer) behind a single class the Flask app can call.

Design goal: this should run out of the box with just scikit-learn
and pandas installed. If the optional heavy libraries below are also
installed, the engine automatically upgrades each component to use
them instead of the lightweight fallback -- mirroring the "safe
initialization with fallback handling" pattern from the notebook's
NewsBot2IntegratedSystem, so one missing dependency never breaks the
whole app.

Optional upgrades (install any/all of these for higher-quality results):
    pip install spacy && python -m spacy download en_core_web_sm   -> real NER
    pip install transformers torch                                 -> transformer sentiment + BART summarization
    pip install sentence-transformers                               -> embedding-based semantic search
    pip install langdetect                                          -> statistical language detection
"""

import os
import re
import math
import string
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_articles.csv")

# ---------------------------------------------------------------------------
# Optional heavy dependencies. Each is imported defensively so the app still
# runs -- with a lightweight fallback -- if a library isn't installed.
# ---------------------------------------------------------------------------
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 42
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


# Small lexicons used by the lightweight fallbacks. These are only used
# when the corresponding heavy model isn't installed.
_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "positive", "success",
    "successful", "growth", "record", "strong", "optimism", "optimistic", "win",
    "wins", "improve", "improved", "improving", "gain", "gains", "expand",
    "expansion", "confidence", "celebrate", "welcomed", "breakthrough", "boost",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "negative", "failure", "crisis",
    "decline", "cut", "cuts", "layoff", "layoffs", "concern", "concerns",
    "warn", "warns", "warning", "risk", "risks", "controversy", "criticism",
    "delay", "delays", "recall", "lawsuit", "cooling", "slowdown", "shortage",
}

# A tiny stopword-overlap heuristic for the language-detection fallback.
_LANG_MARKERS = {
    "en": {"the", "and", "of", "to", "in", "is", "for", "on", "with", "that"},
    "es": {"el", "la", "los", "las", "de", "que", "en", "y", "un", "una", "con"},
    "fr": {"le", "la", "les", "des", "de", "et", "un", "une", "en", "que", "pour"},
}

ENTITY_LABELS = {"PERSON", "ORG", "GPE", "MONEY", "DATE"}


class NewsBotEngine:
    """
    Unified NewsBot intelligence engine.

    Mirrors the four-module architecture from the NewsBot 2.0 notebook:
      Module A - classify_text / discover_topics / analyze_sentiment / extract_entities
      Module B - summarize_text / semantic_search
      Module C - detect_language
      Module D - answer_query (simple intent-based conversational layer)
    """

    def __init__(self, n_topics=5):
        self.status = {}  # tracks which components are running "full" vs "fallback"
        self.corpus = self._load_corpus()

        self._setup_classifier()
        self._setup_topic_model(n_topics=n_topics)
        self._setup_sentiment()
        self._setup_ner()
        self._setup_language_detection()
        self._setup_summarizer()
        self._setup_semantic_search()

    # ------------------------------------------------------------------
    # Setup helpers -- each wrapped in try/except so one failure can't
    # take down the whole engine (matches NewsBot2IntegratedSystem).
    # ------------------------------------------------------------------
    def _load_corpus(self):
        try:
            df = pd.read_csv(DATA_PATH)
            return df
        except Exception as e:
            print(f"Warning: could not load sample corpus ({e}). Using empty corpus.")
            return pd.DataFrame(columns=["id", "category", "title", "text"])

    def _setup_classifier(self):
        """Module A: TF-IDF + Logistic Regression classifier, trained on the
        bundled sample corpus at startup. Swap in your own trained model
        by replacing self.classifier.model with a fitted pipeline."""
        try:
            self.classifier = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])
            X = self.corpus["text"].tolist()
            y = self.corpus["category"].tolist()
            if len(set(y)) >= 2:
                self.classifier.fit(X, y)
                self.classifier_ready = True
                self.status["classification"] = "ready (trained on bundled sample corpus)"
            else:
                self.classifier_ready = False
                self.status["classification"] = "unavailable (not enough training data)"
        except Exception as e:
            self.classifier_ready = False
            self.status["classification"] = f"unavailable ({e})"

    def _setup_topic_model(self, n_topics=5):
        """Module A: LDA topic discovery over the bundled corpus."""
        try:
            self.topic_vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words="english", max_features=2000)
            doc_term_matrix = self.topic_vectorizer.fit_transform(self.corpus["text"].tolist())
            self.topic_model = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
            self.topic_model.fit(doc_term_matrix)
            self.topic_feature_names = self.topic_vectorizer.get_feature_names_out()
            self.topics_ready = True
            self.status["topic_modeling"] = f"ready ({n_topics} topics discovered from bundled corpus)"
        except Exception as e:
            self.topics_ready = False
            self.status["topic_modeling"] = f"unavailable ({e})"

    def _setup_sentiment(self):
        """Module A: transformer sentiment model if available, else lexicon fallback."""
        self.sentiment_pipeline = None
        if _TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_pipeline = hf_pipeline("sentiment-analysis")
                self.status["sentiment"] = "ready (transformer: distilbert-base-uncased-finetuned-sst-2-english)"
            except Exception as e:
                self.status["sentiment"] = f"fallback (transformer load failed: {e})"
        else:
            self.status["sentiment"] = "fallback (lexicon-based; install 'transformers' + 'torch' for the transformer model)"

    def _setup_ner(self):
        """Module A: spaCy NER if available, else a regex-based fallback."""
        self.nlp = None
        if _SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.status["entities"] = "ready (spaCy en_core_web_sm)"
            except OSError:
                self.status["entities"] = "fallback (run: python -m spacy download en_core_web_sm)"
        else:
            self.status["entities"] = "fallback (regex-based; install 'spacy' + en_core_web_sm for full NER)"

    def _setup_language_detection(self):
        """Module C: langdetect if available, else a stopword-overlap heuristic."""
        self.status["language_detection"] = (
            "ready (langdetect)" if _LANGDETECT_AVAILABLE
            else "fallback (heuristic; install 'langdetect' for statistical detection)"
        )

    def _setup_summarizer(self):
        """Module B: BART summarizer if available, else extractive fallback."""
        self.summarizer_pipeline = None
        if _TRANSFORMERS_AVAILABLE:
            try:
                self.summarizer_pipeline = hf_pipeline("summarization", model="facebook/bart-large-cnn")
                self.status["summarization"] = "ready (facebook/bart-large-cnn)"
            except Exception as e:
                self.status["summarization"] = f"fallback (transformer load failed: {e})"
        else:
            self.status["summarization"] = "fallback (extractive TF-IDF scoring; install 'transformers' + 'torch' for BART)"

    def _setup_semantic_search(self):
        """Module B: sentence-transformer embeddings if available, else TF-IDF cosine similarity."""
        self.embedder = None
        self.corpus_embeddings = None
        try:
            self.search_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            self.corpus_tfidf = self.search_vectorizer.fit_transform(self.corpus["text"].tolist())
        except Exception:
            self.search_vectorizer = None
            self.corpus_tfidf = None

        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                self.corpus_embeddings = self.embedder.encode(self.corpus["text"].tolist())
                self.status["semantic_search"] = "ready (sentence-transformers: all-MiniLM-L6-v2)"
            except Exception as e:
                self.status["semantic_search"] = f"fallback (embedding model load failed: {e})"
        else:
            self.status["semantic_search"] = "fallback (TF-IDF cosine similarity; install 'sentence-transformers' for embeddings)"

    # ------------------------------------------------------------------
    # Module A: classification
    # ------------------------------------------------------------------
    def classify_text(self, text):
        if not self.classifier_ready:
            return {"primary_category": "Unknown", "confidence": 0.0, "alternatives": []}

        probabilities = self.classifier.predict_proba([text])[0]
        classes = self.classifier.classes_
        order = np.argsort(probabilities)[::-1]

        primary = order[0]
        alternatives = [
            {"category": classes[i], "confidence": round(float(probabilities[i]), 4)}
            for i in order[1:4]
        ]
        return {
            "primary_category": classes[primary],
            "confidence": round(float(probabilities[primary]), 4),
            "alternatives": alternatives,
        }

    # ------------------------------------------------------------------
    # Module A: topic modeling
    # ------------------------------------------------------------------
    def get_topics(self, top_n_words=8):
        if not self.topics_ready:
            return []
        topics = []
        for topic_idx, topic in enumerate(self.topic_model.components_):
            top_indices = topic.argsort()[::-1][:top_n_words]
            words = [self.topic_feature_names[i] for i in top_indices]
            topics.append({"topic_id": topic_idx, "top_words": words})
        return topics

    # ------------------------------------------------------------------
    # Module A: sentiment
    # ------------------------------------------------------------------
    def analyze_sentiment(self, text):
        if self.sentiment_pipeline is not None:
            try:
                result = self.sentiment_pipeline(text[:512])[0]
                label = result["label"].capitalize()
                if label.upper() not in ("POSITIVE", "NEGATIVE"):
                    label = "Neutral"
                return {"label": label, "confidence": round(float(result["score"]), 4), "method": "transformer"}
            except Exception:
                pass  # fall through to lexicon fallback

        words = re.findall(r"[a-zA-Z']+", text.lower())
        pos = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
        if pos > neg:
            label, confidence = "Positive", min(0.5 + 0.1 * (pos - neg), 0.95)
        elif neg > pos:
            label, confidence = "Negative", min(0.5 + 0.1 * (neg - pos), 0.95)
        else:
            label, confidence = "Neutral", 0.5
        return {"label": label, "confidence": round(confidence, 4), "method": "lexicon"}

    # ------------------------------------------------------------------
    # Module A: named entity recognition
    # ------------------------------------------------------------------
    def extract_entities(self, text):
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                entities = []
                seen = set()
                for ent in doc.ents:
                    if ent.label_ in ENTITY_LABELS and ent.text not in seen:
                        entities.append({"text": ent.text, "label": ent.label_})
                        seen.add(ent.text)
                return entities
            except Exception:
                pass  # fall through to regex fallback

        # Lightweight fallback: capitalized multi-word phrases, money, and dates.
        entities, seen = [], set()

        money = re.findall(r"\$\s?\d[\d,.]*\s?(?:million|billion|trillion)?", text)
        for m in money:
            if m not in seen:
                entities.append({"text": m.strip(), "label": "MONEY"})
                seen.add(m)

        date_words = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|yesterday|tomorrow|this week|next month|this year|last year|this spring|next year)"
        for d in re.findall(date_words, text, flags=re.IGNORECASE):
            if d not in seen:
                entities.append({"text": d, "label": "DATE"})
                seen.add(d)

        # Sequences of capitalized words not at the start of a sentence-ish position.
        tokens = text.split()
        i = 0
        common_words = {"The", "A", "An", "This", "That", "These", "Those", "It"}
        while i < len(tokens):
            word = tokens[i].strip(string.punctuation)
            if word and word[0:1].isupper() and word not in common_words and word.isalpha():
                phrase = [word]
                j = i + 1
                while j < len(tokens):
                    nxt = tokens[j].strip(string.punctuation)
                    if nxt and nxt[0:1].isupper() and nxt.isalpha():
                        phrase.append(nxt)
                        j += 1
                    else:
                        break
                phrase_text = " ".join(phrase)
                if len(phrase_text) > 2 and phrase_text not in seen:
                    label = "ORG" if len(phrase) > 1 else "PERSON"
                    entities.append({"text": phrase_text, "label": label})
                    seen.add(phrase_text)
                i = j
            else:
                i += 1

        return entities[:15]

    # ------------------------------------------------------------------
    # Module C: language detection
    # ------------------------------------------------------------------
    def detect_language(self, text):
        if _LANGDETECT_AVAILABLE:
            try:
                results = detect_langs(text)
                top = results[0]
                return {"language": top.lang, "confidence": round(float(top.prob), 4), "method": "langdetect"}
            except Exception:
                pass

        words = set(re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower()))
        scores = {lang: len(words & markers) for lang, markers in _LANG_MARKERS.items()}
        best_lang = max(scores, key=scores.get) if any(scores.values()) else "en"
        return {"language": best_lang, "confidence": 0.5, "method": "heuristic"}

    # ------------------------------------------------------------------
    # Module B: summarization
    # ------------------------------------------------------------------
    def summarize_text(self, text, max_sentences=3):
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if s]
        if len(sentences) <= max_sentences:
            return {"summary": text.strip(), "method": "passthrough (already short)"}

        if self.summarizer_pipeline is not None:
            try:
                input_len = len(text.split())
                max_len = max(20, min(130, int(input_len * 0.6)))
                result = self.summarizer_pipeline(text, max_length=max_len, min_length=15, do_sample=False)
                return {"summary": result[0]["summary_text"].strip(), "method": "transformer (BART)"}
            except Exception:
                pass  # fall through to extractive fallback

        # Extractive fallback: score sentences by TF-IDF term weight.
        try:
            vec = TfidfVectorizer(stop_words="english")
            tfidf = vec.fit_transform(sentences)
            scores = tfidf.sum(axis=1).A1
            top_idx = sorted(np.argsort(scores)[::-1][:max_sentences])
            summary = " ".join(sentences[i] for i in top_idx)
            return {"summary": summary, "method": "extractive (TF-IDF sentence scoring)"}
        except Exception:
            return {"summary": " ".join(sentences[:max_sentences]), "method": "extractive (first sentences)"}

    # ------------------------------------------------------------------
    # Module B: semantic search over the bundled corpus
    # ------------------------------------------------------------------
    def semantic_search(self, query, top_k=5):
        if self.corpus.empty:
            return []

        if self.embedder is not None and self.corpus_embeddings is not None:
            try:
                query_vec = self.embedder.encode([query])
                sims = cosine_similarity(query_vec, self.corpus_embeddings)[0]
                method = "embeddings (all-MiniLM-L6-v2)"
            except Exception:
                sims = None
        else:
            sims = None

        if sims is None:
            if self.search_vectorizer is None:
                return []
            query_vec = self.search_vectorizer.transform([query])
            sims = cosine_similarity(query_vec, self.corpus_tfidf)[0]
            method = "TF-IDF cosine similarity"

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in top_indices:
            if sims[i] <= 0:
                continue
            row = self.corpus.iloc[i]
            results.append({
                "title": row["title"],
                "category": row["category"],
                "text": row["text"],
                "score": round(float(sims[i]), 4),
            })
        return {"results": results, "method": method}

    # ------------------------------------------------------------------
    # Module D: simple conversational query layer
    # ------------------------------------------------------------------
    def answer_query(self, query):
        """
        Very small intent classifier over the bundled corpus, mirroring the
        notebook's ConversationalInterface: detect an intent (search /
        sentiment filter / summarize), then act on the bundled corpus.
        """
        q = query.lower().strip()

        # Detect a sentiment filter mentioned in the query.
        sentiment_filter = None
        if any(w in q for w in ["positive", "good news", "upbeat"]):
            sentiment_filter = "Positive"
        elif any(w in q for w in ["negative", "bad news"]):
            sentiment_filter = "Negative"

        # Detect a category mentioned in the query.
        category_filter = None
        for cat in self.corpus["category"].unique():
            if cat.lower() in q:
                category_filter = cat
                break

        intent = "search"
        if q.startswith(("summarize", "summarise")) or "summary" in q:
            intent = "summarize"
        elif "compare" in q:
            intent = "compare"

        if category_filter:
            # A category was named explicitly -- start from every article in
            # that category rather than a narrower semantic top-k, so a
            # sentiment filter applied afterward has the full pool to search.
            subset = self.corpus[self.corpus["category"] == category_filter]
            candidates = [
                {"title": r["title"], "category": r["category"], "text": r["text"], "score": None}
                for _, r in subset.iterrows()
            ]
        else:
            search = self.semantic_search(query, top_k=10)
            candidates = search["results"] if isinstance(search, dict) else []

        if sentiment_filter:
            candidates = [c for c in candidates if self.analyze_sentiment(c["text"])["label"] == sentiment_filter]

        # Fill in a display score for category-only candidates (no semantic score yet).
        for c in candidates:
            if c.get("score") is None:
                c["score"] = 1.0

        candidates = candidates[:5]

        if intent == "summarize" and candidates:
            top = candidates[0]
            summary = self.summarize_text(top["text"])
            return {
                "intent": intent,
                "message": f"Here's a summary of the top match, \"{top['title']}\":",
                "summary": summary["summary"],
                "articles": candidates,
            }

        if not candidates:
            return {
                "intent": intent,
                "message": "I couldn't find any matching articles in the sample corpus. Try a different topic, category, or sentiment.",
                "articles": [],
            }

        return {
            "intent": intent,
            "message": f"Found {len(candidates)} matching article(s):",
            "articles": candidates,
        }

    # ------------------------------------------------------------------
    # Orchestration: full single-article analysis (Module A + B + C)
    # ------------------------------------------------------------------
    def analyze_complete(self, text):
        if not text or len(text.strip()) < 10:
            return {"error": "Please provide at least 10 characters of text to analyze."}

        try:
            classification = self.classify_text(text)
            sentiment = self.analyze_sentiment(text)
            entities = self.extract_entities(text)
            language = self.detect_language(text)
            summary = self.summarize_text(text)

            return {
                "success": True,
                "classification": classification,
                "sentiment": sentiment,
                "entities": entities,
                "language": language,
                "summary": summary,
                "statistics": {
                    "word_count": len(text.split()),
                    "character_count": len(text),
                    "sentence_count": len(re.split(r"(?<=[.!?])\s+", text.strip())),
                },
                "original_text": text,
            }
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
