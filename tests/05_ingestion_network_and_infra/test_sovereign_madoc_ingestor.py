import json
import tempfile
from pathlib import Path
import unittest

from src.data.sovereign_ingestor import SovereignIngestor
from src.data.conversational_types import Conversation


class TestSovereignMadocIngestor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.madoc_path = self.root_path / "madoc"
        self.madoc_path.mkdir(parents=True, exist_ok=True)
        self.ingestor = SovereignIngestor(repository_root=str(self.root_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_madoc_nonexistent_directory(self):
        empty_ingestor = SovereignIngestor(repository_root=str(self.root_path / "nonexistent"))
        result = empty_ingestor.ingest_madoc_snapshot()
        self.assertEqual(result, [])

    def test_madoc_jsonl_threaded_ingestion(self):
        # Create mock MADOC jsonl file with threaded Reddit/Voat posts
        jsonl_file = self.madoc_path / "reddit_sample.jsonl"
        items = [
            # Root post
            {
                "id": "post_100",
                "parent_id": "",
                "platform": "reddit",
                "community": "technology",
                "title": "Decentralized P2P Substrates",
                "content": "Discussing resilient peer-to-peer data ingestion.",
                "author": "alice",
                "created_utc": 1600000000,
                "score": 42
            },
            # Comment replying to root
            {
                "id": "comment_101",
                "parent_id": "post_100",
                "platform": "reddit",
                "community": "technology",
                "title": "",
                "content": "Fuzzy decoding preserves topological friction.",
                "author": "bob",
                "created_utc": 1600000050,
                "score": 15
            },
            # Second reply
            {
                "id": "comment_102",
                "parent_id": "post_100",
                "platform": "reddit",
                "community": "technology",
                "title": "",
                "content": "Option D ensures high entropy across dialogue turns.",
                "author": "charlie",
                "created_utc": 1600000100,
                "score": 8
            }
        ]
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        convs = self.ingestor.ingest_madoc_snapshot(sub_path="madoc")
        self.assertEqual(len(convs), 1)
        conv = convs[0]
        self.assertEqual(conv.conversation_id, "madoc_reddit_post_100")
        self.assertEqual(len(conv.turns), 3)
        self.assertEqual(conv.turns[0].speaker_id, "alice")
        self.assertIn("Decentralized P2P Substrates", conv.turns[0].text)
        self.assertEqual(conv.turns[1].speaker_id, "bob")
        self.assertEqual(conv.turns[2].speaker_id, "charlie")
        self.assertEqual(conv.context["platform"], "reddit")
        self.assertEqual(conv.context["community"], "technology")

    def test_madoc_platform_filter(self):
        jsonl_file = self.madoc_path / "multi_platform.jsonl"
        items = [
            {"id": "b1", "parent_id": "", "platform": "bluesky", "content": "Bluesky post", "author": "u1"},
            {"id": "v1", "parent_id": "", "platform": "voat", "content": "Voat post", "author": "u2"},
            {"id": "k1", "parent_id": "", "platform": "koo", "content": "Koo post", "author": "u3"}
        ]
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        bluesky_convs = self.ingestor.ingest_madoc_snapshot(platform_filter="bluesky")
        self.assertEqual(len(bluesky_convs), 1)
        self.assertEqual(bluesky_convs[0].context["platform"], "bluesky")

        voat_convs = self.ingestor.ingest_madoc_snapshot(platform_filter="voat")
        self.assertEqual(len(voat_convs), 1)
        self.assertEqual(voat_convs[0].context["platform"], "voat")

    def test_madoc_convokit_format_json(self):
        json_file = self.madoc_path / "convokit_formatted.json"
        data = [
            {
                "conversation_id": "madoc_ck_01",
                "turns": [
                    {"speaker_id": "userA", "text": "First turn text", "timestamp": "2026-04-01T12:00:00"},
                    {"speaker_id": "userB", "text": "Second turn response", "timestamp": "2026-04-01T12:05:00"}
                ],
                "context": {"corpus": "madoc_sample"},
                "source": "madoc_snapshot"
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        convs = self.ingestor.ingest_madoc_snapshot()
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].conversation_id, "madoc_ck_01")
        self.assertEqual(len(convs[0].turns), 2)
        self.assertEqual(convs[0].turns[0].speaker_id, "userA")

    def test_madoc_user_explanation_and_direct_file_path(self):
        json_file = self.madoc_path / "single_file.jsonl"
        items = [
            {"id": "sf_1", "parent_id": "", "platform": "reddit", "content": "Direct file content", "author": "dev"}
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        explanation_text = "Provenance verified: Non-infringing topological research ground."
        convs = self.ingestor.ingest_madoc_snapshot(
            sub_path=str(json_file),
            explanation=explanation_text
        )
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].context.get("user_explanation"), explanation_text)
