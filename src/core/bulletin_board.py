import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class BulletinBoard(nn.Module):
    """
    Asynchronous "Bulletin Board" force-exchange mechanism.
    Decouples System 1 (Repaired Residues) from System 2 (Constraint Probing).
    
    Allows independent execution of the residue repair loop and the 
    topological constraint verification loop.
    """
    def __init__(self, size: int, device: str = None, alpha: float = 0.15):
        super().__init__()
        self.size = size
        self.device = device
        self.alpha = alpha
        # Metrics registry for diagnostic reporting
        self.metrics = {}
        # Global Force Register:  F_i
        self.register_buffer('force_register', torch.zeros(size, device=device))
        # Residue Mailbox: Stores the latest 'Corrected' residues from System 1
        self.register_buffer('residue_mailbox', torch.zeros(size, device=device))
        # Timestamp/Lock-free counter to detect stale forces
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long, device=device))
        # Asynchronous residue history for micro-stepping
        self.register_buffer('residue_history', torch.zeros(8, size, device=device))
        self.history_idx = 0
        
    def post_force(self, force: torch.Tensor, alpha: float = None):
        """
        System 2 posts a local constraint force to the board.
        Uses tuned EMA (default alpha=0.15 with shock damping) so System 1
        retains relational persistence before physical constraints overwrite the state.
        """
        # Handle batch dimension by taking the mean
        if force.dim() > 1:
            force = force.mean(dim=0)
            
        force_flat = force.detach().view(-1)
        if force_flat.shape[0] < self.size:
            force_flat = torch.nn.functional.pad(force_flat, (0, self.size - force_flat.shape[0]))
        elif force_flat.shape[0] > self.size:
            force_flat = force_flat[:self.size]

        base_alpha = float(alpha if alpha is not None else self.alpha)
        shock_norm = float(torch.norm(force_flat).item())
        eff_alpha = base_alpha / (1.0 + 0.15 * shock_norm)
        self.force_register.copy_((1.0 - eff_alpha) * self.force_register + eff_alpha * force_flat.to(self.force_register.device))
        self.update_count += 1
        
    def read_force(self) -> torch.Tensor:
        """System 1 reads the aggregated force for the current residue."""
        return self.force_register.clone()
        
    def post_residue(self, residue: torch.Tensor):
        """System 1 posts its current 'repaired' state."""
        # Handle batch dimension by taking the mean
        if residue.dim() > 1:
            residue = residue.mean(dim=0)
            
        # Ensure it is 1D to match buffer shape [size]
        residue_flat = residue.detach().view(-1)
        self.residue_mailbox.copy_(residue_flat)
        # Store in history for micro-stepping
        self.residue_history[self.history_idx].copy_(residue_flat)
        self.history_idx = (self.history_idx + 1) % 8
        
    def read_residue(self) -> torch.Tensor:
        """System 2 reads the latest residue to probe for constraints."""
        return self.residue_mailbox.clone()

    def post_metrics(self, payload: Dict[str, Any]):
        """System 1 or Orchestrator posts diagnostic metrics."""
        self.metrics.update(payload)
        
    def read_metrics(self) -> Dict[str, Any]:
        """Read the latest diagnostic payload."""
        return self.metrics.copy()

    def micro_step(self, dt: float = 0.01):
        """
        Asynchronous micro-step for ADMM consensus.
        Smoothly interpolates forces between major updates to prevent manifold rupture.
        """
        # Pseudo-interpolation of force towards current goal
        target_force = self.read_force()
        self.force_register.add_(target_force * dt) # Euler micro-step
