"""
Textbook Quality Filter — Inspired by Microsoft's phi-1 "Textbooks Are All You Need"

Applies heuristic quality filtering to identify "textbook-quality" training
samples: self-contained, instructive, and algorithmically rich content.

Compliance:
    - No cross-domain scalarization (Invariant Optimization Tripwire 3)
    - Each quality dimension gated independently via per-dimension thresholds
    - Uses admissibility (pass/fail per dimension), not scalar reward
"""

import re
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class QualityReport:
    """Quality assessment for a single training sample.
    
    Each dimension is assessed independently — no cross-domain scalarization.
    Admissibility requires ALL dimensions to pass their respective thresholds.
    """
    is_admissible: bool = False
    self_contained: float = 0.0
    instructive: float = 0.0
    algorithmic: float = 0.0
    clarity: float = 0.0
    structural_honesty: float = 0.0
    dimension_gates: Dict[str, bool] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_admissible': self.is_admissible,
            'self_contained': round(self.self_contained, 3),
            'instructive': round(self.instructive, 3),
            'algorithmic': round(self.algorithmic, 3),
            'clarity': round(self.clarity, 3),
            'structural_honesty': round(self.structural_honesty, 3),
            'dimension_gates': self.dimension_gates,
            'flags': self.flags,
        }


class TextbookFilterConfig:
    """
    Dynamic configuration registry for Textbook quality assessment.
    
    Replaces hardcoded constants with a parameterizable container that allows
    continuous systemic adjustment. No cross-domain scalarization allowed.
    """
    def __init__(self, **kwargs):
        # Parametric thresholds (subject to dynamic regime scaling)
        self.thresholds = {
            'self_contained': kwargs.get('self_contained_threshold', 0.3),
            'instructive': kwargs.get('instructive_threshold', 0.3),
            'algorithmic': kwargs.get('algorithmic_threshold', 0.15),
            'clarity': kwargs.get('clarity_threshold', 0.3),
            'structural_honesty': kwargs.get('structural_honesty_threshold', 0.8),
        }
        
        # Continuous heuristic weights (replacing magic numbers)
        self.weights = {
            'code_import_multiplier': 3.0,
            'code_heavy_dep_penalty': 0.3,
            'code_boilerplate_penalty': 0.5,
            'code_has_docstrings': 0.4,
            'code_has_comments': 0.3,
            'code_has_type_hints': 0.2,
            'code_has_example': 0.1,
            'code_algo_scaling': 0.15,
            'code_func_density_bonus': 0.2,
            'code_class_density_bonus': 0.1,
            'code_math_base': 0.3,
            'code_math_multiplier': 0.1,
            'code_long_line_ratio_penalty': 0.5,
            
            'dishonest_hit_penalty': 0.5,
            
            'inst_qa_bonus': 0.8,
            'inst_non_qa_base': 0.4,
            'inst_too_brief_score': 0.2,
            'inst_too_verbose_score': 0.5,
            'inst_word_count_base': 0.3,
            'inst_word_count_slope': 500.0,
            'inst_word_count_cap': 0.9,
            'inst_struct_bonus': 0.15,
            'inst_algo_scaling': 0.12,
            'inst_low_prose_penalty': 0.5,
            'inst_code_block_bonus': 0.2,
            'inst_multi_block_bonus': 0.3,
            'inst_math_base': 0.3,
            'inst_math_multiplier': 0.1,
            'inst_repetitive_penalty': 0.5,
        }


class TextbookRegistry:
    """
    Manages domain vocabularies and structural patterns dynamically.
    
    Allows the autonomous reasoner or user to inject new contexts and domain-specific
    honesty conventions without editing core engine source.
    """
    def __init__(self):
        self.algorithmic_keywords = set()
        self.boilerplate_keywords = set()
        self.dishonest_patterns = []
        self.heavy_dependency_patterns = []
        self.load_defaults()
        
    def load_defaults(self):
        """Initializes the standard architectural vocabulary defaults."""
        self.algorithmic_keywords = {
            'algorithm', 'sort', 'search', 'tree', 'graph', 'hash', 'queue', 'stack',
            'recursive', 'dynamic programming', 'binary search', 'linked list',
            'complexity', 'O(n)', 'O(log', 'breadth-first', 'depth-first',
            'memoize', 'backtrack', 'greedy', 'divide and conquer',
            'matrix', 'vector', 'tensor', 'gradient', 'optimization',
            'probability', 'statistics', 'regression', 'classification',
            'iterate', 'convergence', 'approximation', 'numerical',
            # Gyroidic / Phase 6 terms
            'gyroid', 'manifold', 'topological', 'residue', 'coprime',
            'polynomial', 'invariant', 'ergodic', 'soliton', 'chirality',
            'fixed-point', 'residue-alignment', 'structural honesty',
            # Advanced Math / Topology terms for ArXiv inclusion
            'persistence', 'persistent', 'homology', 'cohomology', 'morphism', 'functor', 
            'category', 'spectral', 'algebra', 'algebraic', 'geometry', 'geometric', 
            'differential', 'bundle', 'sheaf', 'sheaves', 'isomorphism', 'homomorphism', 
            'contact', 'foliation', 'symplectic', 'module', 'modules', 'group', 'groups', 
            'structure', 'structures', 'space', 'spaces'
        }
        
        self.boilerplate_keywords = {
            'todo', 'fixme', 'hack', 'workaround', 'placeholder',
            'autogenerated', 'auto-generated', 'do not edit',
            'pylint: disable', 'noqa', 'pragma: no cover',
        }
        
        self.dishonest_patterns = [
            r'pass\s*#',          # pass-statement as placeholder
            r'TODO:',             # literal TODO
            r'torch\.randn\(',    # placeholder random weights
            r'fake_data',         # explicit fake data
            r'placeholder',       # literal placeholder
        ]
        
        self.heavy_dependency_patterns = [
            r'from\s+\w+\.vendor\.',
            r'import\s+setuptools',
            r'from\s+_?compat\s+import',
            r'# -\*- coding:',
        ]
        
    def add_algorithmic_term(self, term: str):
        self.algorithmic_keywords.add(term.lower())
        
    def add_boilerplate_term(self, term: str):
        self.boilerplate_keywords.add(term.lower())
        
    def register_dishonest_pattern(self, pattern: str):
        self.dishonest_patterns.append(pattern)


# Global instances for backward-compatible default usage
DEFAULT_CONFIG = TextbookFilterConfig()
DEFAULT_REGISTRY = TextbookRegistry()



class TextbookFilter:
    """
    Heuristic quality classifier that identifies "textbook-quality" content.
    
    Quality is assessed across four INDEPENDENT dimensions, each with its own
    admissibility threshold. No cross-domain scalarization — each dimension
    must independently pass its gate for the sample to be admissible.
    
    Textbook quality means ALL of:
    1. Self-contained: Can be understood without external context
    2. Instructive: Teaches a concept or demonstrates a technique
    3. Algorithmic: Involves meaningful computation or reasoning
    4. Clear: Well-structured with explanatory comments or documentation
    """
    
    def __init__(
        self,
        self_contained_threshold: Optional[float] = None,
        instructive_threshold: Optional[float] = None,
        algorithmic_threshold: Optional[float] = None,
        clarity_threshold: Optional[float] = None,
        honesty_threshold: Optional[float] = None,
        config: Optional[TextbookFilterConfig] = None,
        registry: Optional[TextbookRegistry] = None
    ):
        """
        Initializes the filter, utilizing configurable parameters instead of
        hardcoded primitives. Supports dynamic configuration and vocabulary.
        """
        # Setup parameter containers
        self.config = config if config is not None else TextbookFilterConfig()
        self.registry = registry if registry is not None else DEFAULT_REGISTRY
        
        # Inject manual overrides if provided (backward compatibility)
        if self_contained_threshold is not None:
            self.config.thresholds['self_contained'] = self_contained_threshold
        if instructive_threshold is not None:
            self.config.thresholds['instructive'] = instructive_threshold
        if algorithmic_threshold is not None:
            self.config.thresholds['algorithmic'] = algorithmic_threshold
        if clarity_threshold is not None:
            self.config.thresholds['clarity'] = clarity_threshold
        if honesty_threshold is not None:
            self.config.thresholds['structural_honesty'] = honesty_threshold
            
        # Expose thresholds mapping dynamically bounded to config
        self.thresholds = self.config.thresholds
        
        # Substrate dynamics trackers
        self.base_thresholds = self.thresholds.copy()
        self.last_hunger = 0.0
        self.is_play_mode = False

    def modulate_by_hunger(self, hunger_factor: float):
        """
        Scales filter admissibility thresholds dynamically based on 'manifold hunger'.
        A topological surgery approach: thresholds are shifted independently based
        on substrate demands, preventing cross-domain scalarization collapse.
        
        Rule: High hunger (starvation) -> Relax thresholds linearly up to 30%.
        """
        self.last_hunger = hunger_factor
        scale = max(0.7, 1.0 - (hunger_factor * 0.3)) # Minimum 70% of original threshold
        
        for dim, base_val in self.base_thresholds.items():
            # Structural honesty must not undergo drastic deterioration
            effective_scale = scale if dim != 'structural_honesty' else max(0.9, scale)
            self.thresholds[dim] = base_val * effective_scale

    def modulate_by_regime(self, is_play_mode: bool):
        """
        Modulates thresholds according to Fast Cop / Slow Cop scheduling regimes.
        - Play mode: Relax constraints by 15% to allow marginal/creative concepts.
        - Seriousness mode: Retore strict baseline thresholds.
        """
        self.is_play_mode = is_play_mode
        scale = 0.85 if is_play_mode else 1.0
        
        for dim, base_val in self.base_thresholds.items():
            # Structural honesty remains tight
            effective_scale = scale if dim != 'structural_honesty' else 1.0
            self.thresholds[dim] = base_val * effective_scale

    
    def assess(self, text: str, source: str = '') -> QualityReport:
        """
        Assess the quality of a training sample.
        
        Each dimension is scored independently and gated against its
        own threshold. Admissibility requires ALL dimensions to pass.
        No cross-domain scalarization.
        
        Args:
            text: The text content to assess
            source: Source identifier (e.g., 'github_repos', 'dolfin')
            
        Returns:
            QualityReport with per-dimension gates and admissibility
        """
        report = QualityReport()
        
        if not text or len(text.strip()) < 50:
            report.flags.append('too_short')
            return report
        
        # Detect content type
        is_code = self._is_code(text)
        
        if is_code:
            report = self._assess_code(text, report)
        else:
            report = self._assess_instruction(text, report)
        
        # --- Structural Honesty (Shared) ---
        report = self._assess_structural_honesty(text, report)
        
        # Determine thresholds to gate against (relax for arxiv articles)
        active_thresholds = self.thresholds.copy()
        if source and source.startswith("arxiv"):
            active_thresholds['algorithmic'] = 0.05
            active_thresholds['instructive'] = 0.2
            
        # Per-dimension gating — each dimension must independently pass
        report.dimension_gates = {
            dim: getattr(report, dim) >= threshold
            for dim, threshold in active_thresholds.items()
        }
        
        # Admissible only if ALL dimension gates pass
        report.is_admissible = all(report.dimension_gates.values())
        
        # Diagnostics: append failed dimensions to flags for better traceability
        if not report.is_admissible:
            for dim, passed in report.dimension_gates.items():
                if not passed:
                    report.flags.append(f"failed_{dim}")
        
        return report
    
    def filter_batch(
        self,
        texts: List[str],
        source: str = ''
    ) -> List[Dict[str, Any]]:
        """
        Filter a batch of texts, returning those that pass all dimension gates.
        
        Returns list of dicts with 'text', 'admissible', and 'report' keys.
        """
        results = []
        
        for text in texts:
            report = self.assess(text, source)
            if report.is_admissible:
                results.append({
                    'text': text,
                    'admissible': True,
                    'report': report.to_dict(),
                })
        
        return results
    
    def _is_code(self, text: str) -> bool:
        """Detect if text is source code."""
        code_indicators = 0
        if 'def ' in text:
            code_indicators += 1
        if 'class ' in text:
            code_indicators += 1
        if 'import ' in text:
            code_indicators += 1
        if 'return ' in text:
            code_indicators += 1
        if re.search(r'^\s*(if|for|while|try)\s', text, re.MULTILINE):
            code_indicators += 1
        if text.count('    ') > 3 or text.count('\t') > 3:
            code_indicators += 1
        
        return code_indicators >= 3
    
    def _assess_code(self, text: str, report: QualityReport) -> QualityReport:
        """Assess code quality using configurable textbook criteria."""
        lines = text.split('\n')
        w = self.config.weights
        
        # --- Self-containment ---
        import_count = sum(1 for l in lines if l.strip().startswith(('import ', 'from ')))
        total_lines = len(lines)
        import_ratio = import_count / max(total_lines, 1)
        
        # Check for heavy external dependencies dynamically
        heavy_deps = sum(1 for p in self.registry.heavy_dependency_patterns if re.search(p, text))
        
        import_penalty = import_ratio * w.get('code_import_multiplier', 3.0)
        dep_penalty = heavy_deps * w.get('code_heavy_dep_penalty', 0.3)
        report.self_contained = max(0, 1.0 - import_penalty - dep_penalty)
        
        # Penalize boilerplate
        if any(kw in text.lower() for kw in self.registry.boilerplate_keywords):
            report.self_contained *= w.get('code_boilerplate_penalty', 0.5)
            report.flags.append('boilerplate_detected')
        
        # --- Instructiveness ---
        has_docstrings = '"""' in text or "'''" in text
        has_comments = sum(1 for l in lines if l.strip().startswith('#')) > 2
        has_type_hints = ': ' in text and '->' in text
        
        report.instructive = 0.0
        if has_docstrings:
            report.instructive += w.get('code_has_docstrings', 0.4)
        if has_comments:
            report.instructive += w.get('code_has_comments', 0.3)
        if has_type_hints:
            report.instructive += w.get('code_has_type_hints', 0.2)
        if 'example' in text.lower() or 'usage' in text.lower():
            report.instructive += w.get('code_has_example', 0.1)
        
        # --- Algorithmic richness ---
        algo_hits = sum(1 for kw in self.registry.algorithmic_keywords if kw in text.lower())
        scaling = w.get('code_algo_scaling', 0.15)
        report.algorithmic = min(algo_hits * scaling, 1.0)
        
        # Function/class density
        func_count = text.count('def ')
        class_count = text.count('class ')
        if 2 <= func_count <= 15:
            report.algorithmic = min(report.algorithmic + w.get('code_func_density_bonus', 0.2), 1.0)
        if class_count >= 1:
            report.algorithmic = min(report.algorithmic + w.get('code_class_density_bonus', 0.1), 1.0)
            
        # Mathematical rigor check
        math_hits = sum(1 for p in [r'\$\$', r'\\\[', r'\\begin\{', r'O\(n', r'O\(N', r'O\(log'] if re.search(p, text))
        if math_hits > 0:
            math_base = w.get('code_math_base', 0.3)
            math_mult = w.get('code_math_multiplier', 0.1)
            report.algorithmic = min(report.algorithmic + math_base + math_mult * math_hits, 1.0)
        
        # --- Clarity ---
        long_lines = sum(1 for l in lines if len(l) > 120)
        long_ratio = long_lines / max(total_lines, 1)
        
        size_score = 1.0
        if total_lines < 20:
            size_score = 0.3
        elif total_lines < 50:
            size_score = 0.7
        elif total_lines > 500:
            size_score = max(0.3, 1.0 - (total_lines - 500) / 1000)
        
        long_penalty = long_ratio * w.get('code_long_line_ratio_penalty', 0.5)
        report.clarity = max(0, size_score - long_penalty)
        
        return report

    def _assess_structural_honesty(self, text: str, report: QualityReport) -> QualityReport:
        """Evaluate alignment with GOVERNANCE_ANTI_LOBOTOMY.md §12 dynamically."""
        dishonest_hits = 0
        for pattern in self.registry.dishonest_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                dishonest_hits += 1
                safe_pattern = pattern.replace("\\", "")
                report.flags.append(f"dishonest_pattern_{safe_pattern}")
        
        if dishonest_hits > 0:
            penalty = self.config.weights.get('dishonest_hit_penalty', 0.5)
            report.structural_honesty = max(0.0, 1.0 - dishonest_hits * penalty)
        else:
            report.structural_honesty = 1.0
            
        return report
    
    def _assess_instruction(self, text: str, report: QualityReport) -> QualityReport:
        """Assess instruction content quality using parametric weights."""
        w = self.config.weights
        
        # --- Self-containment ---
        has_qa = any(marker in text.lower() for marker in 
                    ['question:', 'answer:', 'instruction:', 'output:', 'user:', 'assistant:'])
        
        report.self_contained = w.get('inst_qa_bonus', 0.8) if has_qa else w.get('inst_non_qa_base', 0.4)
        
        # --- Instructiveness ---
        word_count = len(text.split())
        
        if word_count < 20:
            report.instructive = w.get('inst_too_brief_score', 0.2)
            report.flags.append('too_brief')
        elif word_count > 2000:
            report.instructive = w.get('inst_too_verbose_score', 0.5)
            report.flags.append('too_verbose')
        else:
            wc_base = w.get('inst_word_count_base', 0.3)
            wc_slope = w.get('inst_word_count_slope', 500.0)
            wc_cap = w.get('inst_word_count_cap', 0.9)
            report.instructive = min(wc_base + word_count / wc_slope, wc_cap)
        
        # Bonus for structured content
        if any(marker in text for marker in ['1.', '2.', '- ', '* ', '```']):
            report.instructive = min(report.instructive + w.get('inst_struct_bonus', 0.15), 1.0)
        
        # --- Algorithmic ---
        algo_hits = sum(1 for kw in self.registry.algorithmic_keywords if kw in text.lower())
        scaling = w.get('inst_algo_scaling', 0.12)
        report.algorithmic = min(algo_hits * scaling, 1.0)
        
        code_blocks = text.count('```')
        if code_blocks > 0:
            code_len = sum(len(c) for c in re.findall(r'```.*?```', text, re.DOTALL))
            prose_len = len(text) - code_len
            
            if prose_len < code_len * 0.2:
                penalty = w.get('inst_low_prose_penalty', 0.5)
                report.instructive *= penalty
                report.clarity *= penalty
                report.flags.append('low_prose_ratio')
            else:
                report.instructive = min(report.instructive + w.get('inst_code_block_bonus', 0.2), 1.0)
                if code_blocks >= 2:
                    report.algorithmic = min(report.algorithmic + w.get('inst_multi_block_bonus', 0.3), 1.0)
                    
        # Mathematical rigor check
        math_hits = sum(1 for p in [r'\$\$', r'\\\[', r'\\begin\{', r'O\(n', r'O\(N', r'O\(log'] if re.search(p, text))
        if math_hits > 0:
            math_base = w.get('inst_math_base', 0.3)
            math_mult = w.get('inst_math_multiplier', 0.1)
            report.algorithmic = min(report.algorithmic + math_base + math_mult * math_hits, 1.0)
        
        # --- Clarity ---
        sentences = text.split('.')
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if 5 < avg_sentence_len < 30:
            report.clarity = 0.8
        elif avg_sentence_len <= 5:
            report.clarity = 0.4
        else:
            report.clarity = 0.5
        
        # Penalize repetition
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                penalty = w.get('inst_repetitive_penalty', 0.5)
                report.clarity *= penalty
                report.flags.append('repetitive')
        
        return report
    
    def get_statistics(self, reports: List[QualityReport]) -> Dict[str, Any]:
        """Compute aggregate statistics from a batch of quality reports."""
        if not reports:
            return {'count': 0}
        
        admissible = [r for r in reports if r.is_admissible]
        
        # Per-dimension pass rates (no cross-domain aggregation)
        dim_pass_rates = {}
        for dim in self.thresholds:
            passing_dim = sum(1 for r in reports if r.dimension_gates.get(dim, False))
            dim_pass_rates[dim] = round(passing_dim / len(reports), 3)
        
        return {
            'count': len(reports),
            'admissible': len(admissible),
            'admissibility_rate': round(len(admissible) / len(reports), 3),
            'dimension_pass_rates': dim_pass_rates,
            'flag_counts': self._count_flags(reports),
        }
    
    def _count_flags(self, reports: List[QualityReport]) -> Dict[str, int]:
        """Count occurrences of each flag."""
        counts = {}
        for r in reports:
            for flag in r.flags:
                counts[flag] = counts.get(flag, 0) + 1
        return counts
