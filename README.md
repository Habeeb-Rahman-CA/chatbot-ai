# Basic AI Chatbot — NLP & AI Interaction

A clean, dependency-free educational AI chatbot built from scratch to demonstrate foundational **Natural Language Processing (NLP)** and **Conversational AI Interaction**.

---

## 📚 Key Concepts You Will Learn

```
                                  THE NLP PIPELINE
                                  
  User Input           Tokenization           Stopword Removal          Stemming
"What is NLP?"  ───►  ["what", "is",   ───►  ["what", "nlp"]    ───►  ["what", "nlp"]
                      "nlp"]                 (filters 'is')
                                                                            │
                                                                            ▼
                                                                     TF-IDF Vector
                                                                     [0.707, 0.707]
                                                                            │
                                                                            ▼
                                                                     Cosine Similarity
                                                                   vs. Intent Patterns
                                                                            │
                                                                            ▼
    Dialogue Memory    ◄───  Template Formatting  ◄─── Predicted Intent: "nlp_explanation"
(stores turn & name)         & Response Generation      (Confidence: 100%)
```

### 1. NLP Fundamentals
- **Tokenization**: Splitting continuous human sentences into discrete words (`tokens`) and stripping punctuation.
- **Stopword Filtering**: Removing high-frequency words (`"the"`, `"is"`, `"at"`) that carry little semantic distinction, while deliberately retaining conversational pronouns (`"my"`, `"your"`).
- **Stemming**: Reducing morphological inflections to base roots (e.g., `"running"` $\rightarrow$ `"run"`, `"cats"` $\rightarrow$ `"cat"`).
- **TF-IDF Vectorization** (Term Frequency - Inverse Document Frequency):
  $$\text{TF}(t, d) = \frac{\text{count}(t, d)}{\text{total tokens in } d}$$
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{count}(d \in D : t \in d)}\right) + 1$$
  Converts text into normalized mathematical coordinate vectors.
- **Cosine Similarity**:
  $$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
  Measures the directional alignment between the user's vector and trained intent vectors.
- **Levenshtein Distance**: Edit distance measuring insertions, deletions, and substitutions to handle human typos (e.g., `"helo"` $\approx$ `"hello"`).

### 2. AI Interaction & Dialogue Management
- **Intent Classification**: Mapping freeform user input into actionable intents (e.g., `greeting`, `time_query`, `joke`).
- **Entity Extraction**: Parsing specific facts from context (e.g., extracting `"Maya"` from `"My name is Maya"`).
- **Multi-turn Context Memory**: Remembering the user's name, previous intent, and turn history across the entire conversation.
- **Confidence Scoring & Fallback Strategy**: Gracefully admitting ignorance and offering guidance when confidence is low ($< 30\%$).

---

## 🚀 Quick Start

### 1. Run Interactive Terminal Chat
```bash
python3 main.py
```

#### In-Chat Commands:
- `/debug` — Toggles the live NLP inspector (displays token lists, intent scores, confidence %).
- `/context` — Views stored conversational memory (turn count, remembered user name).
- `/reset` — Clears conversation memory.
- `/help` — Lists sample questions and features.
- `/quit` — Exits chat.

---

### 2. Run Modern Web Interface
Launch the built-in Flask web server:
```bash
python3 app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
Features a live dark-mode chat UI with a real-time **NLP Inspector Drawer** showing:
- Preprocessed tokens
- Predicted intent with confidence meter
- Matched training pattern
- Contextual state memory

---

### 3. Run Automated Unit Tests
To verify all NLP calculations, stemming rules, and intent predictions:
```bash
python3 -m unittest discover -s tests
```

---

## 📂 Project Architecture

```
chatbot-ai/
├── nlp.py              # Pure-Python NLP engine (tokenization, stemming, TF-IDF, similarity)
├── chatbot.py          # Chatbot brain (intent classification, entity extraction, state tracking)
├── intents.json        # Knowledge base with training patterns, intents, and responses
├── main.py             # Interactive CLI with colorized output and /debug mode
├── app.py              # Flask web server providing REST endpoints
├── templates/
│   └── index.html      # Responsive web UI with real-time NLP inspection drawer
├── tests/
│   └── test_chatbot.py # 13 unit tests covering NLP math and dialogue logic
└── README.md           # Documentation and educational walkthrough
```

