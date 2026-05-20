import unittest

from tee_aanikasikirjoitus import normalise_chat_line_text, apply_emotion_hints_rule_based
from tee_aanikirja import build_dialogue_payload_summary


class ChatNormalisationTests(unittest.TestCase):
    def test_chat_square_brackets_without_colon(self):
        self.assertEqual(
            normalise_chat_line_text("[Bäfäfan99] Vuopunk in real life? Let's gooooo!"),
            "Bäfäfan99: Vuopunk in real life? Let's gooooo!",
        )

    def test_chat_square_brackets_with_colon(self):
        self.assertEqual(
            normalise_chat_line_text("[Bäfäfan99]: Vuopunk in real life?"),
            "Bäfäfan99: Vuopunk in real life?",
        )

    def test_prose_with_brackets_is_unchanged(self):
        prose = "Tämä on [hakasu] normaalia proosaa eikä chat-rivi."
        self.assertEqual(normalise_chat_line_text(prose), prose)


class EmotionHintTests(unittest.TestCase):
    def test_rule_based_allows_angry_and_softly(self):
        text = "No, mikä hätänä?! hän kivahti, mutta lempeni sitten. Ai Kitten, käy sisään."
        hinted = apply_emotion_hints_rule_based(text, enabled=True)
        self.assertIn("[angry]", hinted)
        self.assertIn("[softly]", hinted)


class DialoguePayloadSummaryTests(unittest.TestCase):
    def test_single_voice_summary(self):
        inputs = [{"text": "Hei", "voice_id": "voice_kertoja"}, {"text": "Moi", "voice_id": "voice_kertoja"}]
        s = build_dialogue_payload_summary(inputs, "eleven_v3", "mp3_44100_128", "fi", None, "auto", [])
        self.assertEqual(s["unique_voice_count"], 1)

    def test_polyphonic_summary(self):
        inputs = [{"text": "Hei", "voice_id": "voice_a"}, {"text": "Moi", "voice_id": "voice_b"}]
        s = build_dialogue_payload_summary(inputs, "eleven_v3", "mp3_44100_128", "fi", None, "auto", [])
        self.assertEqual(s["unique_voice_count"], 2)


if __name__ == "__main__":
    unittest.main()
