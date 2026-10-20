"""
Chatbot engine that coordinates NLP processing, intent classification,
entity extraction, conversational state/memory, and response generation.
"""

from __future__ import annotations
import datetime
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nlp import (
    SimpleTfidfVectorizer,
    cosine_similarity,
    preprocess,
    string_similarity,
)


@dataclass
class ConversationContext:
    """Stores multi-turn dialogue state and user variables."""
    user_name: str | None = None
    turn_count: int = 0
    last_intent: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def reset(self) -> None:
        self.user_name = None
        self.turn_count = 0
        self.last_intent = None
        self.history.clear()


@dataclass
class ChatbotResult:
    """Encapsulates the chatbot's response alongside educational NLP metadata."""
    response: str
    intent: str
    confidence: float
    matched_pattern: str | None
    extracted_entities: dict[str, Any]
    tokens: list[str]
    is_fallback: bool


class BasicAIChatbot:
    """
    An educational, rule + vector similarity based AI chatbot.
    Demonstrates:
    - Text tokenization & stemming
    - TF-IDF document embedding
    - Cosine similarity matching
    - Fuzzy string distance for typo tolerance
    - Contextual state memory
    """

    CONFIDENCE_THRESHOLD = 0.30

    def __init__(self, intents_path: str | Path | None = None):
        if intents_path is None:
            intents_path = Path(__file__).parent / "intents.json"
        self.intents_path = Path(intents_path)
        self.context = ConversationContext()
        self.vectorizer = SimpleTfidfVectorizer(filter_stops=True, stem_words=True)

        self.intents: list[dict[str, Any]] = []
        self.fallbacks: list[str] = []
        self.corpus_patterns: list[str] = []
        self.pattern_to_intent: list[dict[str, Any]] = []
        self.pattern_vectors: list[list[float]] = []

        self._load_intents()
        self._train_vectorizer()

    def _load_intents(self) -> None:
        """Load knowledge base intents from JSON."""
        with open(self.intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.intents = data.get("intents", [])
        self.fallbacks = data.get("fallbacks", ["I didn't quite catch that."])

    def _train_vectorizer(self) -> None:
        """
        Extract all training patterns from all intents and fit the TF-IDF vectorizer.
        """
        self.corpus_patterns = []
        self.pattern_to_intent = []

        for intent in self.intents:
            tag = intent["tag"]
            for pattern in intent.get("patterns", []):
                # Normalize template patterns like 'my name is {name}' for training
                clean_pattern = re.sub(r"\{[a-zA-Z_]+\}", "", pattern).strip()
                if not clean_pattern:
                    clean_pattern = pattern
                self.corpus_patterns.append(clean_pattern)
                self.pattern_to_intent.append({
                    "intent": tag,
                    "pattern": pattern,
                    "clean_pattern": clean_pattern,
                    "data": intent
                })

        # Train TF-IDF on all pattern documents
        self.vectorizer.fit(self.corpus_patterns)
        self.pattern_vectors = [
            self.vectorizer.transform(pat) for pat in self.corpus_patterns
        ]

    def extract_entities(self, text: str) -> dict[str, Any]:
        """
        Extract key entities (like user names) using regular expression heuristics.
        """
        entities: dict[str, Any] = {}

        # Heuristic 1: "My name is [Name]", "I am [Name]", "I'm [Name]", "Call me [Name]"
        name_patterns = [
            r"\bmy name is\s+([A-Za-z]+)",
            r"\bi am\s+([A-Za-z]+)",
            r"\bi'm\s+([A-Za-z]+)",
            r"\bcall me\s+([A-Za-z]+)",
            r"\bname is\s+([A-Za-z]+)",
        ]
        for pat in name_patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip().capitalize()
                # Exclude common false positives
                if extracted.lower() not in {"happy", "sad", "good", "fine", "here", "curious", "interested"}:
                    entities["user_name"] = extracted
                    break

        return entities

    def predict_intent(self, text: str) -> tuple[str, float, str | None, dict[str, Any]]:
        """
        Predict user intent using:
        1. Exact / Entity trigger patterns
        2. TF-IDF Cosine Similarity
        3. Fuzzy Levenshtein fallback (for short phrases and typos)
        """
        text_clean = text.strip()
        tokens = preprocess(text_clean, filter_stops=False, stem_words=False)

        # 1. Quick regex/entity check for name declaration
        entities = self.extract_entities(text_clean)
        if "user_name" in entities:
            # Matched name declaration pattern
            for item in self.pattern_to_intent:
                if item["intent"] == "user_name_tell":
                    return "user_name_tell", 1.0, item["pattern"], item["data"]

        # 2. Vector-based Cosine Similarity
        user_vector = self.vectorizer.transform(text_clean)
        best_similarity = 0.0
        best_match_idx = -1

        for i, pat_vec in enumerate(self.pattern_vectors):
            sim = cosine_similarity(user_vector, pat_vec)
            if sim > best_similarity:
                best_similarity = sim
                best_match_idx = i

        # 3. Fuzzy String Similarity (Levenshtein) for short sentences / typos
        # Useful when user types "helo" or "byeee" which TF-IDF might miss
        fuzzy_best_sim = 0.0
        fuzzy_match_idx = -1
        for i, pat in enumerate(self.corpus_patterns):
            f_sim = string_similarity(text_clean, pat)
            if f_sim > fuzzy_best_sim:
                fuzzy_best_sim = f_sim
                fuzzy_match_idx = i

        # Blend scores: give weight to fuzzy if TF-IDF found zero overlap
        if best_similarity < 0.25 and fuzzy_best_sim >= 0.70:
            best_similarity = fuzzy_best_sim
            best_match_idx = fuzzy_match_idx

        if best_match_idx != -1 and best_similarity >= self.CONFIDENCE_THRESHOLD:
            matched_item = self.pattern_to_intent[best_match_idx]
            return (
                matched_item["intent"],
                best_similarity,
                matched_item["pattern"],
                matched_item["data"]
            )

        return "unknown", best_similarity, None, {}

    def respond(self, user_input: str) -> ChatbotResult:
        """
        Full conversation pipeline:
        Input -> Preprocessing -> Entity Extraction -> Intent Prediction -> State Update -> Output
        """
        self.context.turn_count += 1
        tokens = preprocess(user_input, filter_stops=True, stem_words=True)
        entities = self.extract_entities(user_input)

        if "user_name" in entities:
            self.context.user_name = entities["user_name"]

        intent, confidence, matched_pattern, intent_data = self.predict_intent(user_input)
        self.context.last_intent = intent

        # Check for fallback
        if intent == "unknown" or confidence < self.CONFIDENCE_THRESHOLD:
            response_text = random.choice(self.fallbacks)
            is_fallback = True
        else:
            is_fallback = False

            # Special case: user asks for name but hasn't set one yet
            if intent == "user_name_query" and not self.context.user_name:
                response_text = intent_data.get(
                    "fallback_response",
                    "I don't know your name yet! You can tell me by saying 'My name is [Your Name]'."
                )
            else:
                responses = intent_data.get("responses", ["I hear you."])
                raw_response = random.choice(responses)

                # Format response with contextual variables
                now = datetime.datetime.now()
                response_text = raw_response.format(
                    user_name=self.context.user_name or "Friend",
                    current_time=now.strftime("%I:%M %p"),
                    current_date=now.strftime("%A, %B %d, %Y")
                )

        # Store in conversation history
        self.context.history.append({"user": user_input, "bot": response_text})

        return ChatbotResult(
            response=response_text,
            intent=intent,
            confidence=round(confidence, 3),
            matched_pattern=matched_pattern,
            extracted_entities=entities,
            tokens=tokens,
            is_fallback=is_fallback
        )

    def reset_conversation(self) -> None:
        """Clear memory and reset conversation state."""
        self.context.reset()

