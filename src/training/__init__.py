"""Structural adaptation and configuration stabilizing."""

# Define early so circular imports never hit missing attributes
try:
    from .trainer import SpectralStructuralTrainer, StructuralAdaptor, ConstraintDataset, collate_fn
except Exception:
    class SpectralStructuralTrainer:
        pass
    StructuralAdaptor = SpectralStructuralTrainer
    class ConstraintDataset:
        pass
    def collate_fn(*args, **kwargs):
        pass

try:
    from .gdpo_trainer import GDPOSovereigntyAdaptor, GDPOSovereigntyPressureComputer
except Exception:
    GDPOSovereigntyAdaptor = None
    GDPOSovereigntyPressureComputer = None

try:
    from .enhanced_temporal_training import NonLobotomyTemporalModel, NonLobotomyTemporalTrainer
except Exception:
    NonLobotomyTemporalModel = None
    NonLobotomyTemporalTrainer = None

__all__ = [
    'StructuralAdaptor',
    'SpectralStructuralTrainer',
    'ConstraintDataset',
    'collate_fn',
    'GDPOSovereigntyAdaptor',
    'GDPOSovereigntyPressureComputer',
    'NonLobotomyTemporalModel',
    'NonLobotomyTemporalTrainer'
]
