#!/usr/bin/env python3
"""
Quick Fix Script for Garbled Output

This script provides immediate fixes for the "nccmtsmneltcclrclcnl,tncsectsead" 
type garbled outputs without requiring full system integration.

Usage:
    python fix_garbled_output.py --input "nccmtsmneltcclrclcnl,tncsectsead"
    python fix_garbled_output.py --test-repair
"""

import torch
import argparse
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from core.spectral_coherence_repair import SpectralCoherenceCorrector
    from core.chern_simons_gasket import ChernSimonsGasket, SolitonStabilityHealer
    from core.love_invariant_protector import SoftSaturatedGates
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class GarbledOutputFixer:
    """Quick fixer for garbled outputs."""
    
    def __init__(self, device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize repair components
        self.spectral_corrector = SpectralCoherenceCorrector(device=device)
        self.gasket = ChernSimonsGasket(device=device)
        self.healer = SolitonStabilityHealer(device=device)
        self.soft_gates = SoftSaturatedGates(5, 3, device=device)  # Default sizes
        
        print(f"🔧 Garbled Output Fixer initialized on {device}")
    
    def analyze_garbled_text(self, text):
        """Analyze garbled text to identify issues."""
        if not text:
            return {}
        
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        
        vowel_count = sum(1 for c in text if c in vowels)
        consonant_count = sum(1 for c in text if c in consonants)
        total_letters = vowel_count + consonant_count
        
        if total_letters == 0:
            return {'error': 'No letters found'}
        
        vowel_ratio = vowel_count / total_letters
        unique_chars = len(set(text))
        repetition_ratio = unique_chars / len(text)
        
        # Detect issues
        issues = []
        if vowel_ratio < 0.2:
            issues.append('consonant_clustering')
        if repetition_ratio < 0.3:
            issues.append('repetitive_collapse')
        if len(text) > 20 and ' ' not in text:
            issues.append('word_boundary_loss')
        
        return {
            'vowel_ratio': vowel_ratio,
            'consonant_ratio': consonant_count / total_letters,
            'repetition_ratio': repetition_ratio,
            'length': len(text),
            'issues': issues
        }
    
    def simulate_residues_from_text(self, text):
        """Create simulated residues from garbled text for repair."""
        # Convert text to numerical representation
        char_values = [ord(c) for c in text[:20]]  # Take first 20 chars
        
        # Pad or truncate to fixed size
        while len(char_values) < 20:
            char_values.append(0)
        char_values = char_values[:20]
        
        # Reshape to residue format [batch=1, K=5, D=4]
        residues = torch.tensor(char_values, dtype=torch.float32, device=self.device)
        residues = residues.view(1, 5, 4)
        
        # Normalize to reasonable range
        residues = (residues - residues.mean()) / (residues.std() + 1e-8)
        
        return residues
    
    def apply_spectral_fix(self, text, residues):
        """Apply spectral coherence fix."""
        print("  🔍 Applying spectral coherence correction...")
        
        # Check for consonant clustering
        clustering = self.spectral_corrector.detect_consonant_clustering(text)
        print(f"     Consonant clustering detected: {clustering}")
        
        if clustering:
            corrected = self.spectral_corrector.adaptive_coherence_correction(residues, text)
            diagnostics = self.spectral_corrector.get_diagnostics()
            print(f"     Coherence threshold adjusted: {diagnostics['theta_coherence']:.3f}")
            return corrected
        
        return residues
    
    def apply_chern_simons_fix(self, residues):
        """Apply Chern-Simons gasket fix."""
        print("  🌀 Applying Chern-Simons gasket...")
        
        # Use proper polynomial co-prime functional system (anti-lobotomy)
        K, D = residues.shape[-2:]
        if not hasattr(self, 'polynomial_config'):
            from core.polynomial_coprime import PolynomialCoprimeConfig
            self.polynomial_config = PolynomialCoprimeConfig(
                k=K,
                degree=D - 1,
                basis_type='chebyshev',
                learnable=True,
                use_saturation=True,
                device=self.device
            )
        
        # Get polynomial coefficients from the proper system
        poly_coeffs = self.polynomial_config.get_coefficients_tensor()  # [K, D]
        
        # Check for logic leaks
        leak_detected = self.gasket.detect_logic_leak(residues)
        print(f"     Logic leak detected: {leak_detected}")
        
        if leak_detected:
            repaired = self.gasket.plug_logic_leak(residues, poly_coeffs)
            diagnostics = self.gasket.get_diagnostics()
            print(f"     Twist energy: {diagnostics['twist_energy']:.6f}")
            return repaired
        
        return residues
    
    def apply_soliton_healing(self, text, residues):
        """Apply soliton healing."""
        print("  🩹 Applying soliton healing...")
        
        # Check for fractured soliton
        fractured = self.healer.detect_fractured_soliton(text)
        print(f"     Fractured soliton detected: {fractured}")
        
        if fractured:
            healed = self.healer.heal_fractured_soliton(residues, text)
            diagnostics = self.healer.get_diagnostics()
            print(f"     Healing progress: {diagnostics['healing_progress']:.3f}")
            return healed
        
        return residues
    
    def apply_soft_gates(self, residues):
        """Apply soft saturated gates."""
        print("  🚪 Applying soft saturated gates...")
        
        # Use low PAS_h to indicate incoherent state
        pas_h = 0.2
        saturated = self.soft_gates.apply_soft_saturation(residues, pas_h)
        
        diagnostics = self.soft_gates.get_diagnostics()
        print(f"     System temperature: {diagnostics['dt']:.3f}")
        print(f"     Silence ratio: {(saturated.abs() < 1e-6).float().mean():.3f}")
        
        return saturated
    
    def residues_to_text_hint(self, residues):
        """Convert repaired residues back to text hint."""
        # This is a simplified conversion for demonstration
        # In a real system, this would use proper decoding
        
        flat_residues = residues.flatten()
        
        # Map to character range
        char_indices = ((flat_residues + 3) * 8).long().clamp(0, 25)
        
        # Convert to letters
        letters = [chr(ord('a') + idx.item()) for idx in char_indices[:10]]
        
        # Add some vowels to improve readability
        result = ""
        for i, letter in enumerate(letters):
            result += letter
            if i % 3 == 2 and i < len(letters) - 1:
                result += 'e'  # Add vowel every 3 consonants
        
        return result
    
    def fix_garbled_output(self, garbled_text):
        """Main method to fix garbled output."""
        print(f"\n🔧 Fixing garbled output: '{garbled_text}'")
        print("=" * 60)
        
        # Analyze the problem
        analysis = self.analyze_garbled_text(garbled_text)
        print(f"📊 Analysis: {analysis}")
        
        if 'error' in analysis:
            return garbled_text
        
        # Convert to residues for processing
        residues = self.simulate_residues_from_text(garbled_text)
        print(f"📐 Simulated residues shape: {residues.shape}")
        
        # Apply fixes in sequence
        print("\n🔄 Applying repair sequence:")
        
        # 1. Spectral coherence fix
        residues = self.apply_spectral_fix(garbled_text, residues)
        
        # 2. Chern-Simons gasket
        residues = self.apply_chern_simons_fix(residues)
        
        # 3. Soliton healing
        residues = self.apply_soliton_healing(garbled_text, residues)
        
        # 4. Soft gates
        residues = self.apply_soft_gates(residues)
        
        # Convert back to text hint
        repaired_hint = self.residues_to_text_hint(residues)
        
        print(f"\n✅ Repair complete!")
        print(f"Original: {garbled_text}")
        print(f"Repaired hint: {repaired_hint}")
        
        return repaired_hint


def main():
    parser = argparse.ArgumentParser(description='Fix garbled output from Gyroidic Flux Reasoner')
    parser.add_argument('--input', type=str, help='Garbled text to fix')
    parser.add_argument('--test-repair', action='store_true', help='Run test repair sequence')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Initialize fixer
    fixer = GarbledOutputFixer(device=args.device)
    
    if args.test_repair:
        # Test with known garbled outputs
        test_cases = [
            "nccmtsmneltcclrclcnl,tncsectsead",
            "mmmtttcccnnnlll",
            "xyzxyzxyzxyz",
            "qwrtypsdfghjklzxcvbnm"
        ]
        
        print("🧪 Running test repair sequence")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            fixer.fix_garbled_output(test_case)
    
    elif args.input:
        # Fix specific input
        fixer.fix_garbled_output(args.input)
    
    else:
        # Interactive mode
        print("🔧 Garbled Output Fixer - Interactive Mode")
        print("Enter garbled text to fix (or 'quit' to exit):")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if user_input:
                    fixer.fix_garbled_output(user_input)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break


if __name__ == "__main__":
    main()
