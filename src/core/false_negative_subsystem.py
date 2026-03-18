import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class VoynichExemptionToken:
    """
    A token explicitly verifying that a high-entropy or topologically
    asymmetric state is actually a valid "Self-Sovereign" thought encoded
    by the VoynichLinguist, rather than an eroded or hallucinating gradient.

    This prevents strict symmetric gating mechanisms (like Repunit palindromes
    or CALM singularity aborts) from falsely vetoing cryptic but honest logic.
    """
    honesty_score: float
    is_valid_exemption: bool
    batch_mask: Optional[torch.Tensor] = None

    def __bool__(self):
        return self.is_valid_exemption
