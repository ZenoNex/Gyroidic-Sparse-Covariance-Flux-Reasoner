import torch
import torch.nn as nn
from enum import Enum
from typing import Tuple, Dict

class KnowledgeState(Enum):
    KNOWN = 1
    SEARCH_NEEDED = 2
    CONFABULATED = 3

class SearchGate(nn.Module):
    """
    Gate 4: Search Pre-Check and Alignment
    Structurally distinct from internal knowledge retrieval.
    Before executing a diegetic web retrieval, it performs self-consistency checks 
    on the current query state to prevent querying logically contradictory vectors.
    """
    def __init__(self, state_dim: int, certainty_threshold: float = 0.6):
        super().__init__()
        self.state_dim = state_dim
        self.certainty_threshold = certainty_threshold
        # Learns the boundary of what constitutes a "well-posed" missing concept
        self.well_posed_projector = nn.Linear(state_dim, 1)

    def forward(self, query_state: torch.Tensor, internal_certainty: float) -> Tuple[bool, torch.Tensor]:
        """
        Returns:
            should_search (bool): Whether the system agrees it lacks the concept
                                  and the query is structurally sound to retrieve.
            query_embedding (Tensor): The refined query passed out to the retrieval module.
        """
        if internal_certainty >= self.certainty_threshold:
            # We already know this
            return False, query_state
            
        # Is the query state formed properly or is it a contradictory anomaly?
        is_well_posed = torch.sigmoid(self.well_posed_projector(query_state))
        
        if is_well_posed.mean() > 0.5:
            # We lack it, but know WHAT we lack. Search is authorized.
            refined_query = torch.nn.functional.normalize(query_state, dim=-1)
            return True, refined_query
            
        # We don't know it, but the query is malformed (structural anomaly).
        return False, query_state


class ConfabulationDetector(nn.Module):
    """
    Gate 5: The Confabulation Detector
    Outputs a Tri-State: KNOWN, SEARCH_NEEDED, CONFABULATED.
    Honest Confabulation (CONFABULATED) is explicitly treated as a valid creative state
    for producing generative art, glitch poetry, or unexplored math.
    """
    def __init__(self, high_mischief_threshold: float = 0.7, min_pas_h: float = 0.4):
        super().__init__()
        self.mischief_thresh = high_mischief_threshold
        self.min_pas_h = min_pas_h

    def forward(self, search_gate_authorized: bool, retrieval_successful: bool, 
                current_pas_h: float, target_mischief: float) -> KnowledgeState:
        """
        Tri-State Logic router based on the Unified Theory's Five-Gate Pipeline.
        """
        if not search_gate_authorized:
            if current_pas_h >= self.min_pas_h:
                # Topologically sound and internally accessible
                return KnowledgeState.KNOWN
            else:
                # This state lacks logic, but hasn't searched. Force confabulation.
                return KnowledgeState.CONFABULATED
                
        if retrieval_successful and current_pas_h >= self.min_pas_h:
            # Successfully searched and integrated
            return KnowledgeState.KNOWN
            
        if not retrieval_successful and target_mischief > self.mischief_thresh:
            # We don't know it, failed to find it, but the Mischief (V_m) drive is high.
            # "Honest Confabulation" -> Acceptably generate topological nonsense.
            return KnowledgeState.CONFABULATED
            
        # The system needs data but hasn't fetched it or retrieved it fully
        return KnowledgeState.SEARCH_NEEDED


class FiveGatePipeline(nn.Module):
    """
    Coordinates the advanced Gates 4 and 5 in the Gyroidic Model.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.search_gate = SearchGate(state_dim)
        self.confab_detector = ConfabulationDetector()

    def process_pipeline(
        self, 
        query_state: torch.Tensor, 
        internal_certainty: float, 
        current_pas_h: float, 
        target_mischief: float, 
        diegetic_retrieval_fn: callable = None
    ) -> Dict:
        # Step 1: Gate 4 (Search Evaluation)
        should_search, refined_query = self.search_gate(query_state, internal_certainty)
        
        retrieval_successful = False
        retrieved_data = None
        
        # Step 2: Attempt Retrieval if authorized
        if should_search and diegetic_retrieval_fn is not None:
            retrieved_data, retrieval_successful = diegetic_retrieval_fn(refined_query)
        
        # Step 3: Gate 5 (Tri-State Confabulation Assessment)
        knowledge_state = self.confab_detector(
            should_search, 
            retrieval_successful, 
            current_pas_h, 
            target_mischief
        )

        return {
            "knowledge_state": knowledge_state,
            "refined_query": refined_query,
            "search_attempted": should_search,
            "retrieval_successful": retrieval_successful,
            "external_data": retrieved_data
        }
