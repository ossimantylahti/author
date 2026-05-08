import unittest

from tee_aanikirja import (
    PronunciationRule,
    build_openai_pronunciation_map,
    preview_pronunciation_rules,
)


class PronunciationPreviewRegressionTests(unittest.TestCase):
    def test_openai_pronunciation_rule_preview_does_not_unpack_crash(self):
        dictionary = {
            "entries": [
                {
                    "graphemes": ["OpenAI"],
                    "origin_language": "en",
                    "pronunciations": {
                        "fi-FI": {
                            "openai_alias": "oupen ai",
                            "instruction": "Lausu englanniksi.",
                        }
                    },
                }
            ]
        }

        active_map = build_openai_pronunciation_map(dictionary, "fi-FI")

        self.assertGreater(len(active_map), 0)
        self.assertTrue(all(isinstance(rule, PronunciationRule) for rule in active_map))

        preview = preview_pronunciation_rules(active_map, limit=20)

        self.assertTrue(preview)
        self.assertIn("OpenAI -> oupen ai", preview[0])


if __name__ == "__main__":
    unittest.main()
