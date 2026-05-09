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
                            "openai_alias": "EH-li-as KOR-pe-la",
                            "instruction": "Pronounce Elias Korpela as Finnish: EH-li-as KOR-pe-la. Korpela has stress on KOR.",
                            "openai_force_alias": "EH-li-ass KOR-pe-la",
                        }
                    },
                },
                {
                    "graphemes": ["Elias"],
                    "origin_language": "fi-FI",
                    "pronunciations": {
                        "fi-FI": {
                            "openai_alias": "E-li-as",
                            "instruction": "Lausu Elias suomalaisena nimenä.",
                            "openai_force_alias": "E-li-ass",
                        },
                        "en-GB": {
                            "openai_alias": "EH-li-as",
                            "instruction": "Pronounce Elias as a Finnish name with three short syllables.",
                            "openai_force_alias": "EH-li-ass",
                        },
                    },
                },
            ]
        }

    def test_finnish_fragment_elias_override_instruction(self):
        text = "Elias katsoi näyttöä."
        rules = build_openai_pronunciation_map(self.dictionary, "fi-FI")
        _, hits = apply_pronunciation_aliases_with_hits(text, rules)
        hits = apply_openai_pronunciation_overrides(text, "fi-FI", hits)
        instruction = build_openai_pronunciation_instruction("fi-FI", hits)
        self.assertTrue(any(h.get("original") == "Elias" for h in hits))
        self.assertIn("Kolme hyvin lyhyttä tavua", instruction)
        self.assertIn("Paino ensimmäisellä tavulla", instruction)
        self.assertIn("Älä lausu nimeä englanniksi", instruction)

    def test_longer_rule_wins_for_elias_korpela(self):
        text = "Elias Korpela opened the stream."
        rules = build_openai_pronunciation_map(self.dictionary, "en-GB")
        rewritten, hits = apply_pronunciation_aliases_with_hits(text, rules)
        hits = apply_openai_pronunciation_overrides(text, "en-GB", hits)
        instruction = build_openai_pronunciation_instruction("en-GB", hits)
        self.assertTrue(rewritten.startswith("EH-li-ass KOR-pe-la"))
        self.assertIn("Finnish proper names in this fragment must keep Finnish pronunciation", instruction)
        self.assertIn("KOR-pe-la", instruction)

    def test_english_fragment_still_gets_elias_override(self):
        text = "You are free now, Elias."
        rules = build_openai_pronunciation_map(self.dictionary, "en-GB")
        _, hits = apply_pronunciation_aliases_with_hits(text, rules)
        hits = apply_openai_pronunciation_overrides(text, "en-GB", hits)
        instruction = build_openai_pronunciation_instruction("en-GB", hits)
        self.assertIn("ignore the default English pronunciation", instruction)
        self.assertIn("Three very short syllables", instruction)
        self.assertIn("Never say ee-LY-us", instruction)


if __name__ == "__main__":
    unittest.main()
