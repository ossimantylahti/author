import unittest

from tee_aanikirja import (
    apply_openai_pronunciation_overrides,
    apply_pronunciation_aliases_with_hits,
    build_openai_pronunciation_instruction,
    build_openai_pronunciation_map,
)


class EliasOpenAIOverrideTests(unittest.TestCase):
    def setUp(self):
        self.dictionary = {
            "entries": [
                {
                    "graphemes": ["Elias Korpela"],
                    "origin_language": "fi-FI",
                    "pronunciations": {
                        "en-GB": {
                            "openai_alias": None,
                            "openai_use_alias": False,
                            "instruction": 'Pronounce "Elias Korpela" as a Finnish name. Elias is close to EL-yahs, not spelled out. Korpela is close to KOR-peh-lah, with stress at the start.',
                        }
                    },
                },
                {
                    "graphemes": ["Elias"],
                    "origin_language": "fi-FI",
                    "pronunciations": {
                        "fi-FI": {
                            "openai_alias": None,
                            "openai_use_alias": False,
                            "instruction": "",
                        },
                        "en-GB": {
                            "openai_alias": None,
                            "openai_use_alias": False,
                            "instruction": 'Pronounce "Elias" as a Finnish proper name, close to "EL-yahs", with stress at the start. Do not spell it out.',
                        },
                    },
                },
            ]
        }

    def test_finnish_fragment_keeps_plain_elias_without_override(self):
        text = "Elias katsoi näyttöä."
        rules = build_openai_pronunciation_map(self.dictionary, "fi-FI")
        rewritten, hits = apply_pronunciation_aliases_with_hits(text, rules)
        hits = apply_openai_pronunciation_overrides(text, "fi-FI", hits)
        instruction = build_openai_pronunciation_instruction("fi-FI", hits)
        self.assertIn("Elias", rewritten)
        self.assertNotIn("E-li-as", rewritten)
        self.assertNotIn("EH-li-as", rewritten)
        self.assertFalse(any(h.get("original") == "Elias" and h.get("instruction") for h in hits))
        self.assertNotIn('Pronounce "Elias" as a Finnish proper name', instruction)

    def test_english_full_name_no_hyphenated_alias_in_text(self):
        text = "Elias Korpela opened the stream."
        rules = build_openai_pronunciation_map(self.dictionary, "en-GB")
        rewritten, hits = apply_pronunciation_aliases_with_hits(text, rules)
        hits = apply_openai_pronunciation_overrides(text, "en-GB", hits)
        instruction = build_openai_pronunciation_instruction("en-GB", hits)
        self.assertIn("Elias Korpela", rewritten)
        self.assertNotIn("EH-li-as", rewritten)
        self.assertNotIn("KOR-pe-la", rewritten)
        self.assertIn('Pronounce "Elias Korpela" as a Finnish name', instruction)

    def test_english_fragment_adds_natural_instruction_without_alias(self):
        text = "You are free now, Elias."
        rules = build_openai_pronunciation_map(self.dictionary, "en-GB")
        rewritten, hits = apply_pronunciation_aliases_with_hits(text, rules)
        hits = apply_openai_pronunciation_overrides(text, "en-GB", hits)
        instruction = build_openai_pronunciation_instruction("en-GB", hits)
        self.assertIn("Elias", rewritten)
        self.assertNotIn("E-li-as", rewritten)
        self.assertNotIn("EH-li-as", rewritten)
        self.assertIn('Pronounce "Elias" as a Finnish proper name', instruction)
        self.assertIn("Do not spell it out", instruction)


if __name__ == "__main__":
    unittest.main()
