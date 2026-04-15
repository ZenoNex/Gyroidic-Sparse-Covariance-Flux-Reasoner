import os
import torch
from pathlib import Path
from src.core.false_negative_subsystem import VoynichExemptionToken

class ArchaeologyRecord:
    """
    Implements the persistent "Carrier Bag" of the Sovereign Engine.
    When an Option D Nutrient is parsed and fossilized, it is saved 
    here. This ensures the system does not start from a sterile blank
    slate, maintaining its "Scars of Interaction" directly on disk.
    """
    def __init__(self, save_dir: str = "archaeology"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.fossil_index = 0
        
    def fossilize(self, token: VoynichExemptionToken) -> bool:
        """
        Takes an Organ of Agency token and commits its state to history.
        Returns True if a scar was saved.
        """
        if token.fossilized_state is not None:
            fossil_path = self.save_dir / f"scar_{self.fossil_index}.pt"
            torch.save(token.fossilized_state.cpu(), fossil_path)
            self.fossil_index += 1
            return True
        return False
