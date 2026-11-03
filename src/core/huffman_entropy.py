"""
Canonical Huffman Entropy Codec and Variable-Length Prefix Bitstream Engine.

Implements information-theoretic entropy coding grounded in Shannon source coding:
    H(X) <= L_avg < H(X) + 1
where H(X) = -sum p_i log2(p_i) is the empirical Shannon entropy.

Used across:
1. PointerlessOctree (Morton delta and material run-length bitstreams).
2. Cleanroom Mechanics (RedNet 16-channel bundled cable and composite conduit telemetry).
3. Knowledge Dyad Fossilizer (Agent Smith mesh payload serialization).
4. Universal Topology Ingestion (Empirical byte entropy estimation and compact prefix bitstreams).
"""

import heapq
import math
from typing import Dict, List, Tuple, Any, Optional, Union
import torch


class HuffmanNode:
    """Binary tree node for Huffman prefix tree construction."""

    def __init__(
        self,
        symbol: Optional[int] = None,
        frequency: int = 0,
        left: Optional["HuffmanNode"] = None,
        right: Optional["HuffmanNode"] = None
    ):
        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __lt__(self, other: "HuffmanNode") -> bool:
        if self.frequency != other.frequency:
            return self.frequency < other.frequency
        # Tie breaker: compare symbols if leaf, else 0
        sym_self = self.symbol if self.symbol is not None else -1
        sym_other = other.symbol if other.symbol is not None else -1
        return sym_self < sym_other


class HuffmanEntropyCodec:
    """
    High-performance Canonical Huffman Codec with Run-Length Bit Packing.
    
    Provides:
    - Optimal prefix code table generation from empirical symbol frequencies.
    - Exact Shannon entropy calculation H(X) and coding efficiency ratio.
    - Bit-level packing into compact bytearrays.
    - Fast streaming decoding.
    - Run-length delta preprocessing for sparse integer streams (e.g. Morton octrees).
    """

    @staticmethod
    def compute_symbol_frequencies(symbols: Union[List[int], bytes, bytearray]) -> Dict[int, int]:
        """Compute frequency distribution of an integer symbol stream or bytes."""
        freqs: Dict[int, int] = {}
        for s in symbols:
            val = int(s)
            freqs[val] = freqs.get(val, 0) + 1
        return freqs

    @staticmethod
    def compute_shannon_entropy(frequencies: Dict[int, int]) -> float:
        """
        Compute empirical Shannon entropy H(X) in bits per symbol:
            H(X) = - sum_{s} p(s) * log2(p(s))
        """
        total = sum(frequencies.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for count in frequencies.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return float(entropy)

    @classmethod
    def build_huffman_tree(cls, frequencies: Dict[int, int]) -> Optional[HuffmanNode]:
        """Build binary Huffman tree using a min-heap priority queue."""
        if not frequencies:
            return None

        # Edge case: single symbol stream
        if len(frequencies) == 1:
            sym, freq = next(iter(frequencies.items()))
            root = HuffmanNode(frequency=freq)
            root.left = HuffmanNode(symbol=sym, frequency=freq)
            return root

        heap: List[HuffmanNode] = []
        for sym, freq in frequencies.items():
            heapq.heappush(heap, HuffmanNode(symbol=sym, frequency=freq))

        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            parent = HuffmanNode(
                symbol=None,
                frequency=node1.frequency + node2.frequency,
                left=node1,
                right=node2
            )
            heapq.heappush(heap, parent)

        return heap[0]

    @classmethod
    def generate_prefix_codes(cls, root: Optional[HuffmanNode]) -> Dict[int, str]:
        """Traverse tree to generate prefix bitstring codes (e.g., '0110')."""
        if root is None:
            return {}

        codes: Dict[int, str] = {}

        def _traverse(node: HuffmanNode, current_code: str):
            if node.is_leaf():
                codes[node.symbol if node.symbol is not None else 0] = current_code or "0"
                return
            if node.left:
                _traverse(node.left, current_code + "0")
            if node.right:
                _traverse(node.right, current_code + "1")

        _traverse(root, "")
        return codes

    @classmethod
    def encode_symbols(cls, symbols: List[int]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compresses an integer symbol sequence into a Huffman-coded bitstream.
        
        Returns:
            packed_bytes: Raw compressed payload bytes.
            metadata: Code dictionary, bit length, Shannon entropy, and efficiency.
        """
        if not symbols:
            return b"", {
                "total_symbols": 0,
                "total_bits": 0,
                "shannon_entropy": 0.0,
                "avg_code_length": 0.0,
                "code_table": {}
            }

        freqs = cls.compute_symbol_frequencies(symbols)
        entropy = cls.compute_shannon_entropy(freqs)
        tree = cls.build_huffman_tree(freqs)
        code_table = cls.generate_prefix_codes(tree)

        # Build bitstream
        bit_chunks: List[str] = [code_table[s] for s in symbols]
        full_bitstring = "".join(bit_chunks)
        total_bits = len(full_bitstring)

        # Pad to byte boundary
        padding = (8 - (total_bits % 8)) % 8
        padded_bits = full_bitstring + ("0" * padding)

        # Pack into bytearray
        packed = bytearray()
        for i in range(0, len(padded_bits), 8):
            byte_val = int(padded_bits[i:i + 8], 2)
            packed.append(byte_val)

        total_symbols = len(symbols)
        avg_code_length = total_bits / total_symbols if total_symbols > 0 else 0.0

        metadata = {
            "total_symbols": total_symbols,
            "total_bits": total_bits,
            "padding": padding,
            "shannon_entropy": entropy,
            "avg_code_length": avg_code_length,
            "coding_efficiency": (entropy / avg_code_length) if avg_code_length > 0 else 1.0,
            # Stringified keys for JSON/PT safety
            "code_table": {str(sym): code for sym, code in code_table.items()}
        }

        return bytes(packed), metadata

    @classmethod
    def decode_symbols(cls, packed_bytes: bytes, metadata: Dict[str, Any]) -> List[int]:
        """Decompresses packed Huffman bytes back to the original symbol sequence."""
        total_symbols = metadata.get("total_symbols", 0)
        total_bits = metadata.get("total_bits", 0)
        if total_symbols == 0 or total_bits == 0:
            return []

        # Invert code table: bitstring -> symbol
        code_table = metadata.get("code_table", {})
        reverse_table: Dict[str, int] = {code: int(sym) for sym, code in code_table.items()}

        # Unpack bits
        bit_list: List[str] = []
        for b in packed_bytes:
            bit_list.append(f"{b:08b}")
        all_bits = "".join(bit_list)[:total_bits]

        decoded: List[int] = []
        current_chunk = ""
        for bit in all_bits:
            current_chunk += bit
            if current_chunk in reverse_table:
                decoded.append(reverse_table[current_chunk])
                current_chunk = ""
                if len(decoded) == total_symbols:
                    break

        return decoded

    @classmethod
    def encode_rle_huffman(cls, data: List[int]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Run-Length Preprocessing + Huffman Coding:
        Transforms runs of identical symbols (e.g. air voxels in an octree)
        into (count, value) pairs and encodes both using dual or interleaved symbol streams.
        """
        if not data:
            return b"", {"type": "rle_huffman", "total_symbols": 0}

        # Compute RLE tuples
        runs: List[int] = []
        curr_val = data[0]
        curr_len = 1

        for val in data[1:]:
            if val == curr_val and curr_len < 255:
                curr_len += 1
            else:
                runs.extend([curr_len, curr_val])
                curr_val = val
                curr_len = 1
        runs.extend([curr_len, curr_val])

        # Encode RLE stream with Huffman
        packed_bytes, meta = cls.encode_symbols(runs)
        meta["type"] = "rle_huffman"
        meta["raw_length"] = len(data)
        meta["rle_pair_count"] = len(runs) // 2
        return packed_bytes, meta

    @classmethod
    def decode_rle_huffman(cls, packed_bytes: bytes, metadata: Dict[str, Any]) -> List[int]:
        """Decompresses an RLE-Huffman bitstream back into the original uncompressed stream."""
        runs = cls.decode_symbols(packed_bytes, metadata)
        original: List[int] = []
        for i in range(0, len(runs), 2):
            if i + 1 < len(runs):
                length = runs[i]
                val = runs[i + 1]
                original.extend([val] * length)
        return original

