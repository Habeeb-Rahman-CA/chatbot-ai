"""
Unit tests for the NLP engine and Chatbot system.
"""

import unittest
from nlp import (
    tokenize,
    remove_stopwords,
    simple_stem,
    preprocess,
    SimpleTfidfVectorizer,
    cosine_similarity,
    levenshtein_distance,
    string_similarity,
)
from chatbot import BasicAIChatbot


class TestNLPEngine(unittest.TestCase):
    def test_tokenize(self):
        text = "Hello, world! How's it going?"
        tokens = tokenize(text)
        self.assertEqual(tokens, ["hello", "world", "how", "s", "it", "going"])

    def test_remove_stopwords(self):
        tokens = ["what", "is", "the", "time"]
        filtered = remove_stopwords(tokens)
        self.assertEqual(filtered, ["what", "time"])

    def test_simple_stem(self):
        self.assertEqual(simple_stem("running"), "run")
        self.assertEqual(simple_stem("cats"), "cat")
        self.assertEqual(simple_stem("happily"), "happi")

    def test_preprocess_pipeline(self):
        text = "The quick brown foxes are jumping"
        result = preprocess(text, filter_stops=True, stem_words=True)
        self.assertIn("quick", result)
        self.assertIn("brown", result)
        self.assertIn("fox", result)
        self.assertIn("jump", result)
        self.assertNotIn("the", result)
        self.assertNotIn("are", result)

    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("hello", "hello"), 0)
        self.assertEqual(levenshtein_distance("helo", "hello"), 1)

    def test_string_similarity(self):
        self.assertAlmostEqual(string_similarity("hello", "hello"), 1.0)
        self.assertGreater(string_similarity("helo", "hello"), 0.7)
        self.assertLess(string_similarity("apple", "banana"), 0.5)

    def test_tfidf_and_cosine_similarity(self):
        docs = [
            "the quick brown fox",
            "jumped over the lazy dog",
            "the cat in the hat"
        ]
        vectorizer = SimpleTfidfVectorizer(filter_stops=True, stem_words=True)
        vectorizer.fit(docs)

        vec1 = vectorizer.transform("brown fox")
        vec2 = vectorizer.transform("quick brown fox")
        vec3 = vectorizer.transform("lazy dog")

        sim_related = cosine_similarity(vec1, vec2)
        sim_unrelated = cosine_similarity(vec1, vec3)

        self.assertGreater(sim_related, 0.5)
        self.assertAlmostEqual(sim_unrelated, 0.0)


class TestChatbot(unittest.TestCase):
    def setUp(self):
        self.bot = BasicAIChatbot()

    def test_greeting(self):
        res = self.bot.respond("Hello!")
        self.assertEqual(res.intent, "greeting")
        self.assertFalse(res.is_fallback)

    def test_entity_extraction_and_name_memory(self):
        # Tell name
        res1 = self.bot.respond("My name is Charlie")
        self.assertEqual(res1.intent, "user_name_tell")
        self.assertEqual(self.bot.context.user_name, "Charlie")
        self.assertIn("Charlie", res1.response)

        # Ask name
        res2 = self.bot.respond("What is my name?")
        self.assertEqual(res2.intent, "user_name_query")
        self.assertIn("Charlie", res2.response)

    def test_bot_identity(self):
        res = self.bot.respond("What is your name?")
        self.assertEqual(res.intent, "bot_identity")
        self.assertIn("PyBot", res.response)

    def test_nlp_explanation(self):
        res = self.bot.respond("What is NLP and how does it work?")
        self.assertEqual(res.intent, "nlp_explanation")

    def test_joke(self):
        res = self.bot.respond("Tell me a funny joke")
        self.assertEqual(res.intent, "joke")

    def test_reset(self):
        self.bot.respond("My name is Bob")
        self.assertEqual(self.bot.context.user_name, "Bob")
        self.bot.reset_conversation()
        self.assertIsNone(self.bot.context.user_name)
        self.assertEqual(self.bot.context.turn_count, 0)


if __name__ == "__main__":
    unittest.main()

