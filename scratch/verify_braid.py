import torch
from src.core.zeitgeist_router import ZeitgeistRouter

def test_braid_relation():
    router = ZeitgeistRouter(dim=256, moduli=(2, 3, 5, 7, 11))
    
    # Test sigma_1 * sigma_2 * sigma_1
    word = [1, 2, 1]
    reduced = router.braid_reduce(word)
    print(f"Braid Word [1, 2, 1] reduced to: {reduced}")
    
    # Test sigma_2 * sigma_1 * sigma_2
    word2 = [2, 1, 2]
    reduced2 = router.braid_reduce(word2)
    print(f"Braid Word [2, 1, 2] reduced to: {reduced2}")
    
    if reduced == [2, 1, 2] or reduced2 == [1, 2, 1]:
         print("SUCCESS: Braid Group relation (sigma_1 sigma_2 sigma_1 = sigma_2 sigma_1 sigma_2) stabilized.")
    else:
         print("FAILURE: Braid Group relation not recognized.")

if __name__ == "__main__":
    test_braid_relation()
