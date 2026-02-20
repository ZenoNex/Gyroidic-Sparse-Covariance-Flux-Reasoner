"""
Local Data Loader — Scan and load datasets from data/raw/ without requiring
any external API tokens or internet access.

Provides a unified interface for loading heterogeneous local data formats
(JSON, JSONL, CSV, Python source files) into training-compatible samples.

Compliance:
    - No hardcoded quality scores (no teleological scalar rewards)
    - Quality assessment deferred to TextbookFilter (per-dimension gating)
    - Structural metadata only (format, source, completeness indicators)
"""

import os
import json
import csv
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field


@dataclass
class DatasetInfo:
    """Metadata about a discovered local dataset."""
    name: str
    path: str
    format: str            # 'jsonl', 'json', 'csv', 'python', 'mixed'
    file_count: int = 0
    total_size_bytes: int = 0
    sample_count: int = 0  # -1 if unknown without full scan
    description: str = ""
    
    @property
    def size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'path': self.path,
            'format': self.format,
            'file_count': self.file_count,
            'total_size_mb': round(self.size_mb, 2),
            'sample_count': self.sample_count,
            'description': self.description,
        }


@dataclass
class TrainingSample:
    """A single training sample in a unified format.
    
    No scalar quality_score — quality is assessed by TextbookFilter
    using per-dimension admissibility gates, not teleological rewards.
    """
    text: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# Dataset-specific descriptions
KNOWN_DATASETS = {
    'HelpSteer': 'NVIDIA reward model annotations with quality ratings (helpfulness, correctness, coherence)',
    'counsel_chat': 'Therapy and counseling Q&A conversations for empathetic response training',
    'dolma': 'OLMo pretraining corpus (URLs/scripts — requires download)',
    'github_repos': 'Curated Python repositories for code understanding and generation',
    'liuhaotianLLaVA-Instruct-150K': 'Multimodal instruction data with text-image pairs for vision-language alignment',
    'quixiAI.dolfin': 'FLAN-based instruction tuning data (alpaca-uncensored, ShareGPT format)',
    'webglm': 'Web-grounded question answering dataset',
}


class LocalDataLoader:
    """
    Scans data/raw/ for available datasets and provides generators
    for loading training samples without requiring HuggingFace tokens.
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Auto-detect project root
            project_root = Path(__file__).parent.parent.parent
            data_dir = str(project_root / 'data' / 'raw')
        self.data_dir = Path(data_dir)
        self._datasets: Dict[str, DatasetInfo] = {}
    
    def scan(self) -> List[DatasetInfo]:
        """Scan data/raw/ and return info about available datasets."""
        self._datasets.clear()
        
        if not self.data_dir.exists():
            return []
        
        for entry in sorted(self.data_dir.iterdir()):
            if not entry.is_dir():
                continue
            
            name = entry.name
            info = self._analyze_directory(name, entry)
            if info.file_count > 0 or info.total_size_bytes > 0:
                self._datasets[name] = info
        
        return list(self._datasets.values())
    
    def _analyze_directory(self, name: str, path: Path) -> DatasetInfo:
        """Analyze a dataset directory for content type and size."""
        extensions = {}
        total_size = 0
        file_count = 0
        
        for f in path.rglob('*'):
            if f.is_file():
                ext = f.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
                total_size += f.stat().st_size
                file_count += 1
        
        # Determine format
        if extensions.get('.jsonl', 0) > 0:
            fmt = 'jsonl'
        elif extensions.get('.json', 0) > 0:
            fmt = 'json'
        elif extensions.get('.csv', 0) > 0:
            fmt = 'csv'
        elif extensions.get('.py', 0) > 0:
            fmt = 'python'
        else:
            fmt = 'mixed'
        
        description = KNOWN_DATASETS.get(name, f'Local dataset ({fmt})')
        
        return DatasetInfo(
            name=name,
            path=str(path),
            format=fmt,
            file_count=file_count,
            total_size_bytes=total_size,
            sample_count=-1,
            description=description,
        )
    
    def get_dataset_info(self, name: str) -> Optional[DatasetInfo]:
        """Get info for a specific dataset."""
        if not self._datasets:
            self.scan()
        return self._datasets.get(name)
    
    def load_samples(
        self,
        dataset_name: str,
        max_samples: int = 1000,
        filter_fn=None
    ) -> Generator[TrainingSample, None, None]:
        """
        Load training samples from a local dataset.
        
        Args:
            dataset_name: Name of the dataset directory in data/raw/
            max_samples: Maximum samples to yield
            filter_fn: Optional callable(sample) -> bool for custom filtering
        
        Yields:
            TrainingSample objects
        """
        if not self._datasets:
            self.scan()
        
        info = self._datasets.get(dataset_name)
        if info is None:
            return
        
        count = 0
        path = Path(info.path)
        
        # Dispatch to format-specific loaders
        if dataset_name == 'counsel_chat':
            gen = self._load_counsel_chat(path)
        elif dataset_name == 'HelpSteer':
            gen = self._load_helpsteer(path)
        elif dataset_name == 'github_repos':
            gen = self._load_github_repos(path)
        elif dataset_name.startswith('quixiAI'):
            gen = self._load_dolfin(path)
        elif dataset_name.startswith('liuhaotian'):
            gen = self._load_llava(path)
        elif dataset_name == 'webglm':
            gen = self._load_webglm(path)
        else:
            gen = self._load_generic(path)
        
        for sample in gen:
            if filter_fn and not filter_fn(sample):
                continue
            
            yield sample
            count += 1
            if count >= max_samples:
                return
    
    # ----------------------------------------------------------------
    # Format-specific loaders
    # ----------------------------------------------------------------
    
    def _load_counsel_chat(self, path: Path) -> Generator[TrainingSample, None, None]:
        """Load therapy Q&A from counsel_chat CSVs."""
        for csv_file in path.glob('*.csv'):
            try:
                with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        question = row.get('questionTitle', row.get('questionText', ''))
                        answer = row.get('answerText', '')
                        if question and answer:
                            text = f"Question: {question}\nAnswer: {answer}"
                            yield TrainingSample(
                                text=text,
                                source='counsel_chat',
                                metadata={
                                    'topic': row.get('topic', 'unknown'),
                                    'has_qa': True,
                                    'format': 'csv_qa',
                                }
                            )
            except Exception:
                continue
        
        # Also load JSON format if present
        for json_file in path.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item.get('text', str(item))
                        yield TrainingSample(
                            text=text,
                            source='counsel_chat',
                            metadata={'format': 'json'},
                        )
            except Exception:
                continue
    
    def _load_helpsteer(self, path: Path) -> Generator[TrainingSample, None, None]:
        """Load HelpSteer with quality annotations."""
        for gz_file in path.glob('*.jsonl.gz'):
            import gzip
            try:
                with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                    for line in f:
                        obj = json.loads(line)
                        prompt = obj.get('prompt', '')
                        response = obj.get('response', '')
                        # HelpSteer has helpfulness, correctness, coherence, complexity, verbosity
                        helpfulness = obj.get('helpfulness', 0)
                        correctness = obj.get('correctness', 0)
                        coherence = obj.get('coherence', 0)
                        quality = (helpfulness + correctness + coherence) / 12.0  # Normalize to 0-1
                        
                        if prompt and response:
                            yield TrainingSample(
                                text=f"User: {prompt}\nAssistant: {response}",
                                source='HelpSteer',
                                metadata={
                                    'helpfulness': helpfulness,
                                    'correctness': correctness,
                                    'coherence': coherence,
                                    'has_qa': True,
                                    'format': 'helpsteer',
                                }
                            )
            except Exception:
                continue
    
    def _load_github_repos(self, path: Path) -> Generator[TrainingSample, None, None]:
        """Load Python source files from cloned repos."""
        for py_file in path.rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8', errors='replace')
                # Skip tiny files and __init__.py
                if len(content) < 100 or py_file.name == '__init__.py':
                    continue
                
                # Heuristic quality: has docstrings + functions + reasonable length
                has_docstring = '"""' in content or "'''" in content
                has_functions = 'def ' in content
                has_classes = 'class ' in content
                line_count = content.count('\n')
                
                quality = 0.3  # Base
                if has_docstring:
                    quality += 0.2
                if has_functions:
                    quality += 0.15
                if has_classes:
                    quality += 0.1
                if 50 < line_count < 500:
                    quality += 0.15
                if 'import' in content and 'return' in content:
                    quality += 0.1
                
                repo_name = py_file.relative_to(path).parts[0] if py_file.relative_to(path).parts else 'unknown'
                
                yield TrainingSample(
                    text=content,
                    source=f'github_repos/{repo_name}',
                    metadata={
                        'filename': py_file.name,
                        'repo': repo_name,
                        'lines': line_count,
                        'has_docstring': has_docstring,
                        'has_functions': has_functions,
                        'has_classes': has_classes,
                        'format': 'python_source',
                    }
                )
            except Exception:
                continue
    
    def _load_dolfin(self, path: Path) -> Generator[TrainingSample, None, None]:
        """Load FLAN/Alpaca instruction data from quixiAI.dolfin."""
        # Prefer deduped versions, smaller files first
        priority_files = [
            'flan1m-alpaca-uncensored-deduped.jsonl',
            'flan1m-sharegpt-deduped.json',
        ]
        
        for fname in priority_files:
            fpath = path / fname
            if not fpath.exists():
                continue
            
            try:
                if fname.endswith('.jsonl'):
                    with open(fpath, 'r', encoding='utf-8') as f:
                        for line in f:
                            obj = json.loads(line)
                            # Alpaca format: instruction, input, output
                            instruction = obj.get('instruction', '')
                            inp = obj.get('input', '')
                            output = obj.get('output', '')
                            
                            if instruction and output:
                                text = f"Instruction: {instruction}"
                                if inp:
                                    text += f"\nInput: {inp}"
                                text += f"\nOutput: {output}"
                                
                                yield TrainingSample(
                                    text=text,
                                    source='dolfin',
                                    metadata={
                                        'format': 'alpaca',
                                        'has_qa': True,
                                    }
                                )
                
                elif fname.endswith('.json'):
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            conversations = item.get('conversations', [])
                            if conversations:
                                text = '\n'.join(
                                    f"{turn.get('from', 'user')}: {turn.get('value', '')}"
                                    for turn in conversations
                                )
                                yield TrainingSample(
                                    text=text,
                                    source='dolfin',
                                    metadata={
                                        'format': 'sharegpt',
                                        'has_qa': True,
                                    }
                                )
            except Exception:
                continue
    
    def _load_llava(self, path: Path) -> Generator[TrainingSample, None, None]:
        """Load LLaVA multimodal instruction data (text portions)."""
        # Start with the smallest file
        for json_file in sorted(path.glob('*.json'), key=lambda f: f.stat().st_size):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    continue
                
                for item in data:
                    conversations = item.get('conversations', [])
                    image = item.get('image', None)
                    
                    text_parts = []
                    for turn in conversations:
                        value = turn.get('value', '')
                        role = turn.get('from', 'human')
                        # Strip image tokens
                        value = value.replace('<image>', '').replace('\n<image>', '').strip()
                        if value:
                            text_parts.append(f"{role}: {value}")
                    
                    if text_parts:
                        yield TrainingSample(
                            text='\n'.join(text_parts),
                            source='llava',
                            metadata={
                                'has_image': image is not None,
                                'image_ref': image,
                                'source_file': json_file.name,
                                'format': 'llava_multimodal',
                            }
                        )
            except Exception:
                continue
    
    def _load_webglm(self, path: Path) -> Generator[TrainingSample, None, None]:
        """Load WebGLM web-grounded QA data."""
        data_dir = path / 'data'
        search_path = data_dir if data_dir.exists() else path
        
        for json_file in search_path.rglob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                items = data if isinstance(data, list) else [data]
                for item in items:
                    question = item.get('question', item.get('input', ''))
                    answer = item.get('answer', item.get('output', ''))
                    if question and answer:
                        yield TrainingSample(
                            text=f"Question: {question}\nAnswer: {answer}",
                            source='webglm',
                            metadata={
                                'grounded': True,
                                'has_qa': True,
                                'format': 'webglm_qa',
                            }
                        )
            except Exception:
                continue
    
    def _load_generic(self, path: Path) -> Generator[TrainingSample, None, None]:
        """Fallback loader for unknown formats."""
        # Try JSONL
        for jsonl_file in path.rglob('*.jsonl'):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        obj = json.loads(line)
                        text = obj.get('text', obj.get('content', str(obj)))
                        yield TrainingSample(
                            text=text,
                            source=path.name,
                            metadata={'format': 'jsonl'},
                        )
            except Exception:
                continue
        
        # Try JSON
        for json_file in path.rglob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    text = item.get('text', item.get('content', str(item)))
                    yield TrainingSample(
                        text=text,
                        source=path.name,
                        metadata={'format': 'json'},
                    )
            except Exception:
                continue
    
    # ----------------------------------------------------------------
    # Batch loading
    # ----------------------------------------------------------------
    
    def load_batch(
        self,
        dataset_name: str,
        batch_size: int = 32,
        max_samples: int = 1000,
    ) -> List[TrainingSample]:
        """Load a batch of samples into a list."""
        samples = []
        for sample in self.load_samples(dataset_name, max_samples):
            samples.append(sample)
            if len(samples) >= batch_size:
                break
        return samples
    
    def load_all_local(
        self,
        max_per_dataset: int = 500,
    ) -> List[TrainingSample]:
        """Load samples from ALL available local datasets."""
        if not self._datasets:
            self.scan()
        
        all_samples = []
        for name in self._datasets:
            for sample in self.load_samples(name, max_per_dataset):
                all_samples.append(sample)
        
        return all_samples
    
    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of all available local datasets."""
        if not self._datasets:
            self.scan()
        
        return {
            'total_datasets': len(self._datasets),
            'total_size_mb': round(sum(d.size_mb for d in self._datasets.values()), 2),
            'datasets': {name: info.to_dict() for name, info in self._datasets.items()},
        }
