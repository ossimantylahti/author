"""Regression tests for the optional Astra model selector."""
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock

# Importing editoi initialises an SDK client but does not make API requests.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")

import editoi


class AstraModelSelectionTests(unittest.TestCase):
    def test_default_model_remains_sol(self):
        self.assertEqual(editoi.DEFAULT_MODEL, "gpt-5.6-sol")
        self.assertNotIn("gpt-6.1-astra", editoi.MODEL_FALLBACKS)
        self.assertNotIn("gpt-6.2-astra", editoi.MODEL_FALLBACKS)

    def test_both_astra_models_are_selectable(self):
        for model in ("gpt-6.1-astra", "gpt-6.2-astra"):
            with self.subTest(model=model):
                self.assertIn(model, editoi.SELECTABLE_MODELS)
                self.assertEqual(
                    editoi._parse_cli_args([f"--model={model}", "book.txt"]),
                    (model, ["book.txt"]),
                )
                self.assertTrue(editoi._supports_explicit_cache(model))

    def test_picker_selects_astra_61_and_62(self):
        with mock.patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.6-sol"}):
            for number, model in (("3", "gpt-6.1-astra"), ("4", "gpt-6.2-astra")):
                with self.subTest(model=model), mock.patch(
                    "builtins.input", return_value=number
                ):
                    self.assertEqual(
                        editoi._parse_cli_args(["--select-model", "book.txt"]),
                        (model, ["book.txt"]),
                    )

    def test_picker_enter_preserves_environment_model(self):
        with mock.patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.6-terra"}), mock.patch(
            "builtins.input", return_value=""
        ):
            self.assertEqual(
                editoi._parse_cli_args(["--select-model", "book.txt"]),
                ("gpt-5.6-terra", ["book.txt"]),
            )

    def test_picker_is_not_launched_by_default(self):
        with mock.patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.6-sol"}), mock.patch(
            "builtins.input", side_effect=AssertionError("Picker should not run")
        ):
            self.assertEqual(
                editoi._parse_cli_args(["book.txt"]),
                ("gpt-5.6-sol", ["book.txt"]),
            )

    def test_api_not_found_for_preview_model_is_explained_before_fallback(self):
        class ModelNotFound(Exception):
            pass

        fake_client = mock.Mock()

        def retrieve(model_name):
            if model_name == "gpt-6.1-astra":
                raise ModelNotFound("No such model")
            return object()

        fake_client.models.retrieve.side_effect = retrieve
        output = io.StringIO()
        with (
            mock.patch.object(editoi, "MODEL", "gpt-6.1-astra"),
            mock.patch.object(editoi, "ACTIVE_MODEL", None),
            mock.patch.object(editoi, "NotFoundError", ModelNotFound),
            mock.patch.object(editoi, "client", fake_client),
            redirect_stdout(output),
        ):
            editoi._select_startup_model()
            self.assertEqual(editoi.ACTIVE_MODEL, editoi.DEFAULT_MODEL)

        self.assertIn("OpenAI API returned NotFound", output.getvalue())
        self.assertIn("not a documented API model ID", output.getvalue())
        self.assertIn("gpt-6-astra", output.getvalue())

    def test_picker_marks_unpublished_models(self):
        output = io.StringIO()
        with mock.patch("builtins.input", return_value=""), redirect_stdout(output):
            selected = editoi._choose_model("gpt-5.6-sol")
        self.assertEqual(selected, "gpt-5.6-sol")
        self.assertIn("ei julkaistu API:ssa", output.getvalue())

    def test_selected_model_is_first_and_fallbacks_are_unchanged(self):
        with mock.patch.object(editoi, "MODEL", "gpt-6.2-astra"):
            candidates = editoi._candidate_models()
            self.assertEqual(candidates[0], "gpt-6.2-astra")
            self.assertEqual(candidates[1:], editoi.MODEL_FALLBACKS)


if __name__ == "__main__":
    unittest.main()
