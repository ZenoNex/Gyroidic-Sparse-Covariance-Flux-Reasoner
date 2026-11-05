"""Data Ingestion Package for Gyroidic Sparse Covariance Flux Reasoner.

Contains Sovereign Ingestion (ArXiv, HN, Wikipedia), GLTF Gaussian Splats,
Local Datasets, and WebP metadata extractors.
"""

from .gltf_splat_ingestor import GltfSplatIngestionPipeline
from .webp_prompt_extractor import WebPPromptExtractor
from .conversational_types import ConversationTurn, Conversation

_LAZY_EXPORTS = {
    'ArXivSovereignIngestor': ('.knowledge_ingestor', 'ArXivSovereignIngestor'),
    'KnowledgeIngestor': ('.knowledge_ingestor', 'KnowledgeIngestor'),
    'ResonanceLarynx': ('.knowledge_ingestor', 'ResonanceLarynx'),
    'LazarusSoftmax': ('.knowledge_ingestor', 'LazarusSoftmax'),
    'EconomicNewsLinker': ('.economic_news_linker', 'EconomicNewsLinker'),
    'InvestorNewsIngestor': ('.economic_news_linker', 'InvestorNewsIngestor'),
    'ChatGPTFrictionHarvester': ('.chatgpt_friction_harvester', 'ChatGPTFrictionHarvester'),
    'OpenScienceIngestor': ('.open_science_ingestor', 'OpenScienceIngestor'),
    'MinecraftIngestionPipeline': ('.minecraft_ingestor', 'MinecraftIngestionPipeline'),
    'LocalDatasetIngestor': ('.local_dataset_ingestor', 'LocalDatasetIngestor'),
    'run_background_topology_analysis': ('.sterile_data_reruns', 'run_background_topology_analysis'),
}

def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        try:
            import importlib
            mod = importlib.import_module(mod_name, package=__name__)
            attr = getattr(mod, attr_name)
            globals()[name] = attr
            return attr
        except Exception:
            globals()[name] = None
            return None
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

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
