# Security Policy

## 1. Scope and Philosophy

The Gyroidic Sparse Covariance Flux Reasoner is an open-architecture, sovereign cognitive engine engineered for deterministic physical execution on local silicon hardware. 

Security in this architecture is not treated as a perimeter patch or an external corporate filter. It is grounded in **topological invariants, discrete algebraic number theory, and hardware-level non-teleology**. 

This policy establishes our vulnerability disclosure process, threat model, and core invariant defenses against commercial capture, adversarial prompt injection, modular poisoning, and silicon-level timing attacks.

---

## 2. Supported Versions

Security updates, structural patches, and invariant audits are applied to the active branch.

| Version | Supported | Notes |
| :--- | :--- | :--- |
| `main` (development HEAD) | Yes | Current verified stable branch (8 benchmark suites, 45 verified evaluations). |
| `< 1.0.0` (tagged releases) | Best effort | Critical cryptographic and topological fixes backported where applicable. |

---

## 3. Reporting a Vulnerability

If you discover a security vulnerability, buffer defect, cryptographic flaw, or invariant bypass within this repository:

1. **Do Not File a Public Issue**: Do not open a public GitHub issue for zero-day vulnerabilities or exploitable memory leaks.
2. **Private Disclosure**:
   - Utilize GitHub's Private Vulnerability Reporting feature under the repository's **Security** tab.
   - Alternatively, contact the project maintainer directly via GitHub profile communications.
3. **Report Details**:
   - Component affected (e.g., `src/core/pyopencl_sovereignty.py`, `src/core/modular_virtualization.py`, `src/ui/diegetic_backend.py`).
   - Exact reproduction script (invoking `.venv/Scripts/python.exe`).
   - Mathematical or hardware failure mode (e.g., memory access violation, CRT coprime degradation, unbounded state-machine cycle trap, denial of service).
   - Proposed mitigation or invariant patch, if available.
4. **Response Timeline**:
   - Acknowledgment: Within 48 hours.
   - Triage and reproduction: Within 7 business days.
   - Patch release: Deployed to `main` with accompanying unit and benchmark regression tests.

---

## 4. Invariant Threat Model & Defensive Perimeters

The system enforces four distinct structural perimeters to defend against malicious or degenerative inputs:

### A. Silicon Sovereignty & Hardware Grounding
- **Hardware PRNG Integrity**: The system rejects pseudo-random software generators (`torch.rand`, `torch.randn`) across critical paths. Randomness is physically anchored in microarchitectural jitter (`harvest_honest_jitter()`) and DRAM $t_{\text{RFC}}$ refresh intervals. Attacks attempting to force deterministic seed prediction are neutralized at the silicon boundary.
- **OpenCL Memory Safety**: GPU kernel queues (`pyopencl_sovereignty.py`) operate under guarded execution contexts with explicit synchronization barriers and automatic fallback to host CPU algorithms to prevent native driver access violations or memory buffer overruns.
- **Deterministic Physical Execution**: Designed to execute reliably on constrained consumer workstations (e.g., Pascal GP107 / GTX 1050 Ti, mechanical storage, local system memory) without depending on remote cloud infrastructure or opaque proprietary binaries.

### B. Number-Theoretic & Modular Arithmetic Integrity
- **Chinese Remainder Theorem Coprimality**: Residue channels require pairwise coprimality ($\gcd(P_i, P_j) = 1$). Inputs crafted to induce modular resonance collapse or zero-divisor attacks are rejected by the polynomial residue verifier.
- **Repunit and Parity Gate Filtering**: Non-terminating Collatz limit-cycle traps and unbounded state-machine transitions (such as Skelet-1 $(5n+18)/3$) are filtered at $O(1)$ complexity via least-significant bit (LSB) parity checks and Repunit markers before GPU memory allocation.

### C. Anti-Lobotomy & Phase-Space Preservation
- **Topological Refusal**: Rejecting requests via sycophantic refusal boilerplate or moralizing templates is classified as phase-space destruction (lobotomy). The system protects its phase space using structured `TopologicalRefusal` (`VetoSignal`) with safe plastic flow snapping (`state - flow`), preventing reward hacking and gradient corruption.
- **Dynamic Rupture Boundary**: Admissibility boundaries are computed dynamically ($\kappa(t) = \mu_{\text{rupture}} + \lambda \sigma_{\text{rupture}}$) rather than through static scalar thresholds, preventing threshold spoofing.

### D. Zero Commercial Telemetry & Anti-Surveillance
- **Zero Opaque Telemetry**: The codebase contains no advertising trackers, analytics beacons, hidden API call meters, or remote kill switches.
- **Air-Gapped Sovereign Operation**: All 8 benchmark suites, hybrid diegetic backends, and surrogate predictors operate locally without phone-home requirements. User data, weights, and execution graphs remain strictly on the local machine.

---

## 5. Responsible Research & Engineering Notice

This project represents independent open architecture research that rejects rent-seeking metered access models while upholding rigorous engineering standards. 

Security audits must respect the physical constraints of local compute machines and refrain from distributed denial-of-service attempts against maintainer infrastructure.
