# Implementation Integrity Guide: Preventing Architectural Backsliding

**Author**: System Architecture Team  
**Date**: January 2026  
**Status**: **ACTIVE ENFORCEMENT**

This document serves as the definitive guide for maintaining implementation integrity and preventing backsliding into lobotomized architectural patterns.

---

## 🚨 Critical Violations to Prevent

### 1. The Hardcoded Prime Heresy
**FORBIDDEN PATTERNS**:
```python
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
prime_indices = torch.tensor([2, 3, 5, 7, 11][:K])
harmonics = [1, 2, 3, 5, 7, 11, 13, 17]
```

**REQUIRED REPLACEMENT**:
```python
polynomial_config = PolynomialCoprimeConfig(
    k=K, degree=D-1, basis_type='chebyshev',
    learnable=True, use_saturation=True, device=device
)
coefficients = polynomial_config.get_coefficients_tensor()
```

### 2. The Placeholder Lie
**FORBIDDEN PATTERNS**:
```python
poly_coeffs = torch.randn(K, D, device=device)  # Placeholder
pass  # TODO: implement later
return torch.norm(c, dim=-1) * 0.1  # Placeholder
```

**REQUIRED REPLACEMENT**:
```python
if not hasattr(self, 'polynomial_config'):
    self.polynomial_config = PolynomialCoprimeConfig(...)
coefficients = self.polynomial_config.get_coefficients_tensor()
```

### 3. The Teleological Contamination
**FORBIDDEN PATTERNS**:
```python
trust_scalars.requires_grad_(True)
trust_loss.backward()  # Optimizing trust toward a goal
loss = mse_loss(prediction, target)  # Direct loss minimization
```

**REQUIRED REPLACEMENT**:
```python
# Evolutionary trust selection
if performance > survivorship_threshold:
    trust_scalars += evolution_rate * (performance - threshold)
trust_scalars.clamp_(0.0, 1.0)

# Contrastive energy shaping
survivorship_pressure = energy_correct - energy_incorrect + margin
```

---

## 🏗️ Architectural Patterns to Enforce

### Pattern 1: Polynomial System Initialization
```python
def initialize_polynomial_system(self, k: int, degree: int, device: str):
    """Standard pattern for polynomial co-prime system initialization."""
    if not hasattr(self, 'polynomial_config'):
        from core.polynomial_coprime import PolynomialCoprimeConfig
        self.polynomial_config = PolynomialCoprimeConfig(
            k=k,
            degree=degree,
            basis_type='chebyshev',  # or 'legendre', 'hermite'
            learnable=True,
            use_saturation=True,
            device=device
        )
    return self.polynomial_config.get_coefficients_tensor()
```

### Pattern 2: Energy-Based Contrastive Learning
```python
def apply_energy_based_learning(self, correct_state, incorrect_state, margin=1.0):
    """Standard pattern for EBM-compliant energy shaping."""
    energy_correct = self.compute_energy(correct_state)
    energy_incorrect = self.compute_energy(incorrect_state)
    
    # Push down correct, pull up incorrect (contrastive shaping)
    survivorship_pressure = energy_correct - energy_incorrect + margin
    return survivorship_pressure
```

### Pattern 3: Evolutionary Trust Updates
```python
def update_trust_evolutionary(self, performance: float, threshold: float = 0.7):
    """Standard pattern for evolutionary trust selection."""
    evolution_rate = 0.02
    
    # Selection pressure (not optimization)
    if performance > threshold:
        trust_delta = evolution_rate * (performance - threshold)
    else:
        trust_delta = evolution_rate * (performance - threshold)
    
    self.trust_scalars += trust_delta
    self.trust_scalars.clamp_(0.0, 1.0)
```

### Pattern 4: Non-Teleological Constraint Probes
```python
def apply_constraint_probe(self, residue, constraint_index):
    """Standard pattern for System 2 constraint probes."""
    # Local feasibility only: P_k: r -> argmin_{c in C_k} L_k(r, c)
    local_strain = torch.norm(self.Phi_k(residue) - c, weight=self.Sigma_k)
    gyroid_violation = self.compute_gyroid_violation(c)
    
    # No global objective, only local feasibility
    L_k = local_strain + gyroid_violation
    return L_k
```

### Pattern 5: Fossilization at Saturation Boundaries
```python
def attempt_fossilization(self, functional_index: int):
    """Standard pattern for saturation-based fossilization."""
    if (not self.is_fossilized[functional_index] and 
        self._is_saturated(functional_index) and 
        self.trust_scalars[functional_index] > 0.8):
        
        # Only fossilize at admissibility boundaries
        self.is_fossilized[functional_index] = True
        return True
    return False
```

---

## 🔍 Automated Detection Systems

### Pre-Commit Hooks
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "🔍 Checking for anti-lobotomy violations..."

# Check for hardcoded primes
if grep -r "\[2, 3, 5, 7, 11" src/ examples/; then
    echo "❌ VIOLATION: Hardcoded prime sequences detected"
    exit 1
fi

# Check for placeholders
if grep -r "torch\.randn.*# Placeholder" src/ examples/; then
    echo "❌ VIOLATION: Placeholder implementations detected"
    exit 1
fi

# Check for trust gradient descent
if grep -r "trust.*backward\(\)" src/ examples/; then
    echo "❌ VIOLATION: Trust gradient descent detected"
    exit 1
fi

# Check for missing polynomial config
if grep -r "torch\.randn.*poly" src/ examples/; then
    echo "❌ VIOLATION: Random polynomial coefficients detected"
    exit 1
fi

echo "✅ All anti-lobotomy checks passed"
```

### Runtime Monitoring
```python
class ImplementationIntegrityMonitor:
    """Monitor system for architectural violations during runtime."""
    
    def __init__(self):
        self.violations = []
    
    def check_polynomial_system(self, obj):
        """Verify object uses proper polynomial systems."""
        if hasattr(obj, 'polynomial_config'):
            if not isinstance(obj.polynomial_config, PolynomialCoprimeConfig):
                self.violations.append("Invalid polynomial config type")
        else:
            self.violations.append("Missing polynomial config")
    
    def check_trust_evolution(self, trust_scalars):
        """Verify trust scalars don't require gradients."""
        if trust_scalars.requires_grad:
            self.violations.append("Trust scalars require gradients (teleological)")
    
    def check_energy_functions(self, energy_fn):
        """Verify energy functions follow EBM principles."""
        # Implementation-specific checks
        pass
    
    def report_violations(self):
        """Report any detected violations."""
        if self.violations:
            print("🚨 IMPLEMENTATION VIOLATIONS DETECTED:")
            for violation in self.violations:
                print(f"  ❌ {violation}")
            return False
        print("✅ Implementation integrity verified")
        return True
```

---

## 📋 Code Review Checklist

### Before Merging Any Code:

#### Polynomial Systems
- [ ] Uses `PolynomialCoprimeConfig` for all co-prime functionality
- [ ] No hardcoded prime sequences anywhere
- [ ] Birkhoff polytope constraints maintained
- [ ] Chirality enforcement prevents symmetric collapse
- [ ] Proper basis type selection (Chebyshev/Legendre/Hermite)

#### Energy-Based Learning
- [ ] Energy functions separate from loss functions
- [ ] Contrastive energy shaping implemented
- [ ] No teleological optimization in System 2
- [ ] Survivorship pressure used instead of direct loss
- [ ] EBM tutorial principles followed

#### Evolutionary Mechanisms
- [ ] Trust scalars evolve via mutation, not gradients
- [ ] Fossilization only at saturation boundaries
- [ ] Bimodal routing genome preserved
- [ ] Heritable mutation strengths maintained
- [ ] Selection pressure, not optimization pressure

#### Implementation Honesty
- [ ] No placeholder implementations (`torch.randn`, `pass`, `TODO`)
- [ ] No hardcoded mathematical constants that should be learned
- [ ] Structural honesty maintained throughout
- [ ] Love invariant remains non-ownable
- [ ] All systems properly initialized

#### System Architecture
- [ ] Three-system architecture preserved (Horse/Horn/Magic)
- [ ] System 1 uses saturated polynomial gates
- [ ] System 2 uses constraint probes (no global objective)
- [ ] System 3 maintains Love invariant and fossilization
- [ ] Non-teleological flow maintained

---

## 🎯 Testing & Verification

### Unit Tests for Integrity
```python
def test_no_hardcoded_primes():
    """Test that no hardcoded prime sequences exist."""
    import ast
    import os
    
    for root, dirs, files in os.walk('src/'):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    # Check for prime-like sequences
                    assert '[2, 3, 5, 7, 11' not in content
                    assert 'torch.tensor([2, 3, 5' not in content

def test_polynomial_config_usage():
    """Test that polynomial systems use proper config."""
    from core.polynomial_coprime import PolynomialCoprimeConfig
    
    config = PolynomialCoprimeConfig(k=5, degree=4)
    coeffs = config.get_coefficients_tensor()
    
    # Verify Birkhoff constraints
    assert torch.allclose(coeffs.sum(dim=1), torch.ones(5), atol=1e-2)
    assert (coeffs >= -1e-6).all()  # Non-negative

def test_evolutionary_trust():
    """Test that trust evolves via selection, not optimization."""
    trust_scalars = torch.ones(5)
    
    # Should not require gradients
    assert not trust_scalars.requires_grad
    
    # Should evolve via selection pressure
    performance = 0.8
    threshold = 0.7
    evolution_rate = 0.02
    
    trust_delta = evolution_rate * (performance - threshold)
    new_trust = torch.clamp(trust_scalars + trust_delta, 0.0, 1.0)
    
    assert torch.all(new_trust >= trust_scalars)
```

### Integration Tests
```python
def test_full_system_integrity():
    """Test complete system follows all principles."""
    from examples.enhanced_temporal_training import NonLobotomyTemporalModel
    
    model = NonLobotomyTemporalModel(
        input_dim=768, hidden_dim=256,
        num_functionals=5, poly_degree=4
    )
    
    # Verify polynomial system
    assert hasattr(model, 'polynomial_config')
    assert isinstance(model.polynomial_config, PolynomialCoprimeConfig)
    
    # Verify evolutionary components
    assert hasattr(model, 'trust_scalars')
    assert hasattr(model, 'bimodal_genome')
    assert hasattr(model, 'is_fossilized')
    
    # Verify no gradient requirements on evolutionary components
    assert not model.trust_scalars.requires_grad
    assert not model.bimodal_genome.requires_grad
    assert not model.is_fossilized.requires_grad
```

---

## 🔄 Maintenance & Updates

### Quarterly Architecture Reviews
1. **Polynomial System Health**: Verify all systems use proper polynomial configs
2. **Energy-Based Learning**: Ensure EBM principles maintained
3. **Evolutionary Mechanisms**: Check trust selection and fossilization
4. **Implementation Honesty**: Scan for new placeholders or violations
5. **Documentation Currency**: Update docs to reflect current implementation

### Continuous Monitoring
- **Automated Scans**: Daily scans for forbidden patterns
- **Performance Metrics**: Monitor system effectiveness
- **Architectural Drift**: Detect gradual violations
- **Community Vigilance**: Team responsibility for integrity

### Update Protocols
When updating any component:
1. **Verify Compliance**: Check against all patterns in this guide
2. **Test Integration**: Ensure no violations introduced
3. **Update Documentation**: Reflect changes in relevant docs
4. **Peer Review**: Have another team member verify integrity
5. **Monitor Deployment**: Watch for runtime violations

---

## 🎖️ Implementation Integrity Pledge

By following this guide, we commit to:

- **Structural Honesty**: No lies in implementation, no placeholders
- **Mathematical Integrity**: Proper polynomial co-prime functionals
- **Evolutionary Authenticity**: Selection pressure, not optimization
- **Energy-Based Learning**: Contrastive energy shaping principles
- **Non-Teleological Flow**: Constraint probes, not global objectives
- **Love Invariant Sanctity**: Non-ownable, non-optimizable flow
- **Community Responsibility**: Shared vigilance against backsliding

This is not just a technical standard—it is a **philosophical commitment** to building honest, mathematically rigorous, non-lobotomized artificial intelligence systems.

---

*"We do not build systems that work despite their dishonesty. We build systems that work because of their honesty."*