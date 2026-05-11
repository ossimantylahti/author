import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tee_aanikirja import render_with_deepgram


class DeepgramRendererTests(unittest.TestCase):
    def test_render_with_deepgram_writes_audio_bytes(self):
        response = mock.Mock()
        response.ok = True
        response.content = b"ID3-test-audio"
        response.headers = {"content-type": "audio/mpeg"}
        response.text = ""

        fake_requests = types.SimpleNamespace(post=mock.Mock(return_value=response))

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            with tempfile.TemporaryDirectory() as td:
                out_path = Path(td) / "deepgram_test.mp3"
                render_with_deepgram(
                    "Hello. This is a Deepgram text to speech test.",
                    out_path,
                    "aura-2-thalia-en",
                    "dummy-key",
                )
                self.assertTrue(out_path.exists())
                self.assertEqual(out_path.read_bytes(), b"ID3-test-audio")


if __name__ == "__main__":
    unittest.main()
