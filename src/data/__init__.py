"""Data Ingestion Package for Gyroidic Sparse Covariance Flux Reasoner.

Contains Sovereign Ingestion (ArXiv, HN, Wikipedia), GLTF Gaussian Splats,
Local Datasets, and WebP metadata extractors.
"""

from .gltf_splat_ingestor import GltfSplatIngestionPipeline
from .webp_prompt_extractor import WebPPromptExtractor
from .conversational_types import ConversationTurn, Conversation

try:
    from .knowledge_ingestor import ArXivSovereignIngestor, KnowledgeIngestor, ResonanceLarynx, LazarusSoftmax
except ImportError:
    ArXivSovereignIngestor = None
    KnowledgeIngestor = None
    ResonanceLarynx = None
    LazarusSoftmax = None

try:
    from .economic_news_linker import EconomicNewsLinker, InvestorNewsIngestor
except ImportError:
    EconomicNewsLinker = None
    InvestorNewsIngestor = None

try:
    from .chatgpt_friction_harvester import ChatGPTFrictionHarvester
except ImportError:
    ChatGPTFrictionHarvester = None

try:
    from .open_science_ingestor import OpenScienceIngestor
except ImportError:
    OpenScienceIngestor = None

try:
    from .minecraft_ingestor import MinecraftIngestionPipeline
except ImportError:
    MinecraftIngestionPipeline = None

try:
    from .local_dataset_ingestor import LocalDatasetIngestor
except ImportError:
    LocalDatasetIngestor = None

try:
    from .sterile_data_reruns import run_background_topology_analysis
except ImportError:
    run_background_topology_analysis = None

__all__ = [
    'GltfSplatIngestionPipeline',
    'WebPPromptExtractor',
    'ConversationTurn',
    'Conversation',
    'ArXivSovereignIngestor',
    'KnowledgeIngestor',
    'ResonanceLarynx',
    'LazarusSoftmax',
    'EconomicNewsLinker',
    'InvestorNewsIngestor',
    'ChatGPTFrictionHarvester',
    'OpenScienceIngestor',
    'MinecraftIngestionPipeline',
    'LocalDatasetIngestor',
    'run_background_topology_analysis',
]
