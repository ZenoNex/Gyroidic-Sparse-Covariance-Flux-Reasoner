import torch
import torch.nn as nn

class BulletinBoard(nn.Module):
    """
    Asynchronous "Bulletin Board" force-exchange mechanism.
    Decouples System 1 (Repaired Residues) from System 2 (Constraint Probing).
    
    Allows independent execution of the residue repair loop and the 
    topological constraint verification loop.
    """
    def __init__(self, size: int, device: str = None):
        super().__init__()
        self.size = size
        self.device = device
        # Global Force Register: Σ F_i
        self.register_buffer('register_buffer', torch.zeros(size, device=device))
        # Residue Mailbox: Stores the latest 'Corrected' residues from System 1
        self.register_buffer('residue_mailbox', torch.zeros(size, device=device))
        # Timestamp/Lock-free counter to detect stale forces
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long, device=device))
        
    def post_force(self, force: torch.Tensor):
        """System 2 posts a local constraint force to the board."""
        # Use exponential moving average to avoid sudden rupture
        alpha = 0.3
        self.register_buffer.copy_((1.0 - alpha) * self.register_buffer + alpha * force.detach())
        self.update_count += 1
        
    def read_force(self) -> torch.Tensor:
        """System 1 reads the aggregated force for the current residue."""
        return self.register_buffer.clone()
        
    def post_residue(self, residue: torch.Tensor):
        """System 1 posts its current 'repaired' state."""
        self.residue_mailbox.copy_(residue.detach())
        
    def read_residue(self) -> torch.Tensor:
        """System 2 reads the latest residue to probe for constraints."""
        return self.residue_mailbox.clone()
