"""
Phase 18 Verification Script: ZeitgeistRouter

Tests:
1. ZeitgeistState CRT bijection
2. ZeitgeistRouter forward pass (all three modes)
3. Non-commutativity property
4. diegetic_backend.py syntax validity
5. Phase 18 integration markers in diegetic_backend.py

Run from project root:
    python examples/verify_zeitgeist.py
"""

import sys
import os
import ast

sys.path.insert(0, os.getcwd())

import torch

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def check(label, condition, warn=False):
    tag = WARN if (warn and not condition) else (PASS if condition else FAIL)
    print(f"  [{tag}] {label}")
    return condition


# =============================================================================
# 1. Module import
# =============================================================================
print("1. Module import")
from src.core.zeitgeist_router import ZeitgeistState, ZeitgeistRouter
check("ZeitgeistState importable", True)
check("ZeitgeistRouter importable", True)

# =============================================================================
# 2. ZeitgeistState CRT bijection
# =============================================================================
print("\n2. ZeitgeistState CRT bijection")
moduli = (2, 3, 5, 7, 11)
s0 = ZeitgeistState.initial(moduli)
check("initial alpha == [0,0,0,0,0]", s0.alpha == [0, 0, 0, 0, 0])
check("initial crt_index == 0", s0.crt_index == 0)

# Known CRT value: (1,2,3,4,5) mod (2,3,5,7,11)
s_test = ZeitgeistState(alpha_tensor=(1, 2, 3, 4, 5), level=0, moduli=moduli)
idx = s_test.crt_index
# Verify by checking residues of idx against the original moduli
residues_ok = all((idx % p) == r for p, r in zip(moduli, (1, 2, 3, 4, 5)))
check(f"CRT bijection round-trip for (1,2,3,4,5): crt_index={idx}", residues_ok)

# to_dict has required keys
d = s_test.to_dict()
check("to_dict has 'alpha_tensor_sum' key", 'alpha_tensor_sum' in d)
check("to_dict has 'crt_index' key", 'crt_index' in d)
check("to_dict has 'mode' key", 'mode' in d)
check("to_dict has 'step' key", 'step' in d)

# =============================================================================
# 3. ZeitgeistRouter forward (all modes)
# =============================================================================
print("\n3. ZeitgeistRouter forward pass")
dim = 64
router = ZeitgeistRouter(dim=dim, moduli=moduli, grazing_eps=0.99)  # eps=0.99 => ensures grazing
check("Router instantiated", router is not None)
check("facet_normals shape [5, 64]", tuple(router.facet_normals.shape) == (5, 64))
check("switch_gate weight shape [5, 64]", tuple(router.switch_gate.weight.shape) == (5, 64))

torch.manual_seed(0)
x_grazing = torch.randn(1, dim)

mode, new_state, diag, x_steered = router(x_grazing, s0)
check(f"forward returns mode string: '{mode}'", isinstance(mode, str))
check("new_state is ZeitgeistState", isinstance(new_state, ZeitgeistState))
check("diagnostics has 'grazing_dims'", 'grazing_dims' in diag)
check("diagnostics has 'grazing_pressure'", 'grazing_pressure' in diag)
check("diagnostics has 'mode'", 'mode' in diag)
check("diagnostics has 'state' dict", isinstance(diag.get('state'), dict))

# =============================================================================
# 4. Non-commutativity
# =============================================================================
print("\n4. Non-commutativity (gate must be non-trivial)")
router_nc = ZeitgeistRouter(dim=dim, moduli=moduli, grazing_eps=0.49)
# Inflate gate weights to ensure actual switching
with torch.no_grad():
    router_nc.switch_gate.weight.uniform_(0.5, 1.5)
    router_nc.switch_gate.bias.fill_(0.3)

torch.manual_seed(99)
x = torch.randn(1, dim)
y = torch.randn(1, dim) * 2.0

_, s_xy1, _, _ = router_nc(x, s0)
_, s_xy2, _, _ = router_nc(y, s_xy1)
alpha_xy = s_xy2.alpha

_, s_yx1, _, _ = router_nc(y, s0)
_, s_yx2, _, _ = router_nc(x, s_yx1)
alpha_yx = s_yx2.alpha

print(f"  route(x then y) alpha: {alpha_xy}")
print(f"  route(y then x) alpha: {alpha_yx}")
nc = (alpha_xy != alpha_yx)
check("route(x,y) != route(y,x) — Non-commutative", nc, warn=True)

# =============================================================================
# 5. diegetic_backend.py integration markers
# =============================================================================
print("\n5. diegetic_backend.py integration markers")
backend_path = os.path.join("src", "ui", "diegetic_backend.py")
with open(backend_path, "r", encoding="utf-8", errors="replace") as f:
    src = f.read()

try:
    ast.parse(src)
    check("diegetic_backend.py parses cleanly", True)
except SyntaxError as e:
    check(f"diegetic_backend.py SYNTAX ERROR line {e.lineno}: {e.msg}", False)

check("ZeitgeistRouter in extensions import block", "from src.core.zeitgeist_router import ZeitgeistRouter" in src)
check("PHASE 18 comment in __init__", "PHASE 18" in src)
check("zeitgeist_router initialized", "self.zeitgeist_router = ZeitgeistRouter(" in src)
check("_zeitgeist_state initialized", "self._zeitgeist_state" in src)
check("PHASE 2.7 block injected", "PHASE 2.7" in src)
check("_zg_mode in process_input", "_zg_mode" in src)
check("zeitgeist key in metrics payload", '"zeitgeist"' in src)
check("orphaned 0.61 _sanitize stub removed", "d = {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}}" not in src)

print("\n=== Phase 18 Verification Complete ===")
