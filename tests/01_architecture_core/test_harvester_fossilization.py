import os
import json
import shutil
import tempfile
import asyncio
import pytest
import torch
from src.data.chatgpt_friction_harvester import ChatGPTFrictionHarvester, auto_temporal_training_loop

class MockFossilizer:
    def __init__(self):
        self.fossilized = []

    def fossilize(self, dyad, text_embedding, seed_state=None):
        self.fossilized.append((dyad, text_embedding, seed_state))
        return "mock_fossil_filename.pt"

class MockModel:
    def __init__(self):
        self.fossilizer = MockFossilizer()

class MockTrainer:
    def __init__(self):
        self.model = MockModel()
        self.trained_interactions = []

    def train_on_interaction(self, input_tensor, response_tensor, tag_weights=None):
        self.trained_interactions.append((input_tensor, response_tensor, tag_weights))
        return {"temporal_coherence": 0.95}

def test_harvester_extraction_and_fossilization():
    # Create a temporary directory for mock ChatGPT export
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a mock conversations JSON file
        conv_data = [
            {
                "title": "Friction test conversation",
                "mapping": {
                    "root": {
                        "parent": None,
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Hello Akkaris, let's explore non-human AI architectures."]}
                        }
                    },
                    "reply": {
                        "parent": "root",
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["I agree, the non-ergodic structure is fascinating."]}
                        }
                    }
                }
            }
        ]

        with open(os.path.join(temp_dir, "conversations-1.json"), "w", encoding="utf-8") as f:
            json.dump(conv_data, f)

        # Initialize harvester (using dim=64 to match standard CanonicalProjector)
        harvester = ChatGPTFrictionHarvester(export_dir=temp_dir, dim=64)

        # Harvest interactions
        interactions = list(harvester.harvest_interactions())
        assert len(interactions) == 1

        in_tensor, out_tensor, tags, user_text, assistant_text = interactions[0]
        
        # Verify text is yielded correctly
        assert user_text == "Hello Akkaris, let's explore non-human AI architectures."
        assert assistant_text == "I agree, the non-ergodic structure is fascinating."
        
        # Verify tags are extracted correctly
        assert tags.get("is_human_alias") == 1.0  # due to "Akkaris"
        assert tags.get("is_nonhuman_archetype") == 1.0  # due to "non-human" / "archetype"

        # Verify training loop and fossilization works
        trainer = MockTrainer()
        
        # Run async loop synchronously
        asyncio.run(auto_temporal_training_loop(harvester, trainer, delay=0.01))

        # Assert trainer trained on interaction
        assert len(trainer.trained_interactions) == 1
        assert trainer.trained_interactions[0][2] == tags

        # Assert fossilizer fossilized the interaction
        assert len(trainer.model.fossilizer.fossilized) == 1
        dyad, in_emb, out_emb = trainer.model.fossilizer.fossilized[0]
        assert dyad.linguistic_description == user_text
        assert dyad.metadata["response_text"] == assistant_text
        assert "is_human_alias" in dyad.metadata["tags"]
        
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_harvester_extraction_and_fossilization()
    print("ALL HARVESTER TESTS PASSED!")
