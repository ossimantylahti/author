import unittest

from tee_aanikirja import (
    apply_openai_pronunciation_overrides,
    apply_pronunciation_aliases_with_hits,
    build_openai_pronunciation_instruction,
    build_openai_pronunciation_map,
)


class OpenAIFinnishAliasSafetyTests(unittest.TestCase):
    def setUp(self):
        self.dictionary = {
            "entries": [
                {"graphemes": ["Miksu"], "origin_language": "fi-FI", "pronunciations": {"en-GB": {"openai_alias": "MIK-soo", "openai_use_alias": False, "instruction": 'Pronounce "Miksu" as a Finnish nickname, close to "Miksu" with the stress at the start. Keep it compact and natural. Do not split it into separate syllables. Do not spell it out.'}, "fi-FI": {"openai_alias": "MIK-soo", "openai_use_alias": False, "instruction": ""}}},
                {"graphemes": ["Elias"], "origin_language": "fi-FI", "pronunciations": {"en-GB": {"openai_alias": "EH-li-as", "openai_use_alias": False, "instruction": 'Pronounce "Elias" as a Finnish proper name, close to "EL-yahs", with stress at the start. Do not split it into separate syllables. Do not spell it out.'}, "fi-FI": {"openai_alias": "E-li-as", "openai_use_alias": False, "instruction": ""}}},
                {"graphemes": ["Elias Korpela"], "origin_language": "fi-FI", "pronunciations": {"en-GB": {"openai_alias": "EH-li-as KOR-pe-la", "openai_use_alias": False, "instruction": 'Pronounce "Elias Korpela" as a Finnish name.'}}},
                {"graphemes": ["Korpela"], "origin_language": "fi-FI", "pronunciations": {"en-GB": {"openai_alias": "KOR-pe-la", "openai_use_alias": False, "instruction": 'Pronounce "Korpela" as a Finnish surname.'}}},
                {"graphemes": ["Virtanen"], "origin_language": "fi-FI", "pronunciations": {"en-GB": {"openai_alias": "VIR-ta-nen", "openai_use_alias": False, "instruction": 'Pronounce "Virtanen" as a Finnish surname.'}}},
                {"graphemes": ["Otso Soini"], "origin_language": "fi-FI", "pronunciations": {"en-GB": {"openai_alias": "OTS-o SOY-ni", "openai_use_alias": False, "instruction": 'Pronounce "Otso Soini" as a Finnish name.'}}},
            ]
        }

    def test_finnish_miksu(self):
        text = "Miksu katsoi Eliasta."
        rules = build_openai_pronunciation_map(self.dictionary, "fi-FI")
        rewritten, hits = apply_pronunciation_aliases_with_hits(text, rules)
        hits = apply_openai_pronunciation_overrides(text, "fi-FI", hits)
        instruction = build_openai_pronunciation_instruction("fi-FI", hits)
        self.assertIn("Miksu", rewritten)
        self.assertIn("Eliasta", rewritten)
        self.assertNotIn("MIK-soo", rewritten)
        self.assertNotIn("E-li-as", rewritten)
        self.assertNotIn("EH-li-as", rewritten)
        self.assertNotIn('Pronounce "Miksu"', instruction)

    def test_english_miksu(self):
        text = "Miksu looked at Elias."
        rules = build_openai_pronunciation_map(self.dictionary, "en-GB")
        rewritten, hits = apply_pronunciation_aliases_with_hits(text, rules)
        hits = apply_openai_pronunciation_overrides(text, "en-GB", hits)
        instruction = build_openai_pronunciation_instruction("en-GB", hits)
        self.assertIn("Miksu", rewritten)
        self.assertIn("Elias", rewritten)
        self.assertNotIn("MIK-soo", rewritten)
        self.assertNotIn("EH-li-as", rewritten)
        self.assertIn("Do not split it into separate syllables.", instruction)
        self.assertIn("Do not spell it out.", instruction)

    def test_english_full_name(self):
        text = "Miksu sent a message to Elias Korpela."
        rules = build_openai_pronunciation_map(self.dictionary, "en-GB")
        rewritten, hits = apply_pronunciation_aliases_with_hits(text, rules)
        instruction = build_openai_pronunciation_instruction("en-GB", apply_openai_pronunciation_overrides(text, "en-GB", hits))
        self.assertIn("Miksu", rewritten)
        self.assertIn("Elias Korpela", rewritten)
        self.assertNotIn("MIK-soo", rewritten)
        self.assertNotIn("EH-li-as", rewritten)
        self.assertNotIn("KOR-pe-la", rewritten)
        self.assertIn('Pronounce "Elias Korpela"', instruction)

    def test_general_finnish_names_not_hyphenated(self):
        text = "Virtanen called Otso Soini."
        rules = build_openai_pronunciation_map(self.dictionary, "en-GB")
        rewritten, hits = apply_pronunciation_aliases_with_hits(text, rules)
        self.assertIn("Virtanen", rewritten)
        self.assertIn("Otso Soini", rewritten)
        self.assertNotIn("VIR-ta-nen", rewritten)
        self.assertNotIn("OTS-o", rewritten)
        self.assertGreaterEqual(len(hits), 2)


if __name__ == "__main__":
    unittest.main()
