import argparse
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import litteroi


def make_args(source: str, **overrides: object) -> argparse.Namespace:
    values = {
        "file": source,
        "start": None,
        "stop": None,
        "language": "fi",
        "model": litteroi.MODEL,
        "diarize": False,
        "speaker_map": {},
        "speaker_count": 2,
        "context_chars": 1200,
        "srt": False,
        "max_file_mb": 20.0,
        "chunk_seconds": 1200.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class LitteroiYouTubeTests(unittest.TestCase):
    def test_recognises_supported_youtube_urls_only(self) -> None:
        self.assertTrue(litteroi.is_youtube_url("https://www.youtube.com/watch?v=abc"))
        self.assertTrue(litteroi.is_youtube_url("https://youtu.be/abc"))
        self.assertFalse(litteroi.is_youtube_url("https://example.com/watch?v=abc"))
        self.assertFalse(litteroi.is_youtube_url("not-a-url"))

    @mock.patch("litteroi.ffmpeg_available", return_value=True)
    @mock.patch("litteroi.subprocess.run")
    def test_download_uses_python_module_and_audio_arguments(
        self, run_mock: mock.Mock, _ffmpeg_mock: mock.Mock
    ) -> None:
        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            output_path = command[command.index("-o") + 1]
            Path(output_path).write_bytes(b"audio")
            return subprocess.CompletedProcess(command, 0, "Video: title?\n", "")

        run_mock.side_effect = fake_run
        with tempfile.TemporaryDirectory() as temp_dir:
            path, title = litteroi.download_youtube_audio(
                "https://youtu.be/abc", temp_dir
            )

        command = run_mock.call_args.args[0]
        self.assertEqual(command[:3], [litteroi.sys.executable, "-m", "yt_dlp"])
        self.assertIn("-x", command)
        self.assertEqual(command[command.index("--audio-format") + 1], "mp3")
        self.assertEqual(command[-1], "https://youtu.be/abc")
        self.assertTrue(path.endswith("youtube_audio.mp3"))
        self.assertEqual(title, "Video_ title_")

    def test_offsets_multiple_chunks_and_renumbers_srt(self) -> None:
        first = litteroi.add_offset_to_segments(
            [{"start": 1, "end": 2, "text": "Ensimmäinen"}], 0
        )
        second = litteroi.add_offset_to_segments(
            [{"start": 3.5, "end": 7, "text": "Toinen"}], 1200
        )
        srt = litteroi.build_srt(first + second, {}, include_speakers=False)

        self.assertIn("1\n00:00:01,000 --> 00:00:02,000", srt)
        self.assertIn("2\n00:20:03,500 --> 00:20:07,000", srt)
        self.assertEqual(srt.count(" --> "), 2)

    def test_srt_timestamp_supports_more_than_one_hour(self) -> None:
        self.assertEqual(litteroi.format_seconds_for_srt(4504.25), "01:15:04,250")

    def test_long_local_audio_uses_existing_chunk_pipeline_and_keeps_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir, "recording.mp3")
            source.write_bytes(b"original")
            chunks = [
                litteroi.ChunkInfo("chunk1.mp3", 1, 2, 0, 1200),
                litteroi.ChunkInfo("chunk2.mp3", 2, 2, 1200, 1200),
            ]
            responses = [
                {"text": "Ensimmäinen", "segments": [{"start": 1, "end": 2, "text": "Ensimmäinen"}]},
                {"text": "Toinen", "segments": [{"start": 3.5, "end": 7, "text": "Toinen"}]},
            ]
            with (
                mock.patch("litteroi.shutil.which", return_value="/usr/bin/tool"),
                mock.patch("litteroi.get_audio_duration_seconds", return_value=2400),
                mock.patch("litteroi.split_mp3_to_chunks", return_value=chunks) as split_mock,
                mock.patch("litteroi.transcribe_mp3", side_effect=responses) as transcribe_mock,
            ):
                litteroi.run(make_args(str(source), srt=True))

            self.assertTrue(source.exists())
            self.assertEqual(source.read_bytes(), b"original")
            split_mock.assert_called_once()
            self.assertEqual(transcribe_mock.call_count, 2)
            srt = Path(temp_dir, "recording.srt").read_text(encoding="utf-8")
            self.assertIn("2\n00:20:03,500 --> 00:20:07,000", srt)

    def test_youtube_audio_is_cleaned_and_srt_is_automatic(self) -> None:
        downloaded_path: list[str] = []

        def fake_download(_url: str, temp_dir: str) -> tuple[str, str]:
            path = os.path.join(temp_dir, "youtube_audio.mp3")
            Path(path).write_bytes(b"temporary")
            downloaded_path.append(path)
            return path, "Safe title"

        response = {
            "text": "Teksti",
            "segments": [{"start": 0.25, "end": 1.5, "text": "Teksti"}],
        }
        with tempfile.TemporaryDirectory() as output_dir:
            with (
                mock.patch("litteroi.download_youtube_audio", side_effect=fake_download),
                mock.patch("litteroi.shutil.which", return_value="/usr/bin/ffprobe"),
                mock.patch("litteroi.get_audio_duration_seconds", return_value=10),
                mock.patch("litteroi.transcribe_mp3", return_value=response) as transcribe_mock,
                mock.patch("litteroi.os.getcwd", return_value=output_dir),
            ):
                litteroi.run(make_args("https://www.youtube.com/watch?v=abc"))

            self.assertFalse(os.path.exists(downloaded_path[0]))
            self.assertTrue(Path(output_dir, "Safe title.txt").is_file())
            self.assertTrue(Path(output_dir, "Safe title.srt").is_file())
            self.assertTrue(transcribe_mock.call_args.args[4])


if __name__ == "__main__":
    unittest.main()
