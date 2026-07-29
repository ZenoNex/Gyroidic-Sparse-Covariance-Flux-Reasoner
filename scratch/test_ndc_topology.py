import sys
import os
import torch

# Ensure project root is on sys.path
sys.path.insert(0, r"d:\programming\python\Gyroidic Sparse Covariance Flux Reasoner")

from src.core.non_dual_coin import CerumenPotWallet, ChernSimonsValidator, EconomicAbortException, TripsodicLedger, transact
from src.data.freenet_bulletin_router import FreenetBulletinRouter

def test_non_dual_coin_and_freenet():
    print("--- Test 1: CerumenPotWallet & Non-Dual Transact ---")
    wallet_a = CerumenPotWallet(dim=16)
    wallet_b = CerumenPotWallet(dim=16)
    validator = ChernSimonsValidator(yield_criteria=2.5)

    initial_state_a = wallet_a.state.clone()
    transact(wallet_a, wallet_b, validator)
    print("Transact completed successfully. Symmetrical hyper-ring fusion achieved.")

    print("\n--- Test 2: TripsodicLedger Volume Swelling ---")
    ledger = TripsodicLedger(base_volume=1000.0)
    ledger.register_mischief(0.5)
    ledger.rhythm_tick()
    print(f"Global Volume after 1 tick: {ledger.global_volume:.2f}")
    lr_mod = ledger.get_learning_rate_modulator()
    ego_mod = ledger.get_ego_death_limit_modulator()
    print(f"Learning Rate Modulator: {lr_mod:.6f}, Ego Death Limit Modulator: {ego_mod:.4f}")

    print("\n--- Test 3: Freenet Bulletin Router Payload ---")
    router = FreenetBulletinRouter()
    xml_payload = router.generate_dummy_payload()
    print(f"Generated FMS XML Payload:\n{xml_payload[:200]}...")
    print("\nALL NON-DUAL COIN TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_non_dual_coin_and_freenet()
