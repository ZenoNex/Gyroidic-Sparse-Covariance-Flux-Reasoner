import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, r"d:\programming\python\Gyroidic Sparse Covariance Flux Reasoner")

from src.topology.bonfire_network import BonfireNetwork
from src.data.economic_news_linker import EconomicAgentLinker
from src.core.collapse_poisoner import CollapsePathPoisoner

def test_zero_dollar_bet_and_booting():
    print("--- Test 1: Zero-Dollar Prediction & Kelly Bet Allocation ---")
    bonfire = BonfireNetwork(node_id="test_alpha")
    bonfire.add_peer("http://localhost:8002")
    
    # Place accurate zero-dollar bet
    res1 = bonfire.place_zero_dollar_bet("http://localhost:8002", predicted_pas=0.55)
    print(f"Prediction Bet 1 Result: {res1}")
    assert res1["challenge_issued"] == False
    assert res1["booted"] == False

    print("\n--- Test 2: Underperforming Bet & Compute Challenge Trigger ---")
    res2 = bonfire.place_zero_dollar_bet("http://localhost:8002", predicted_pas=0.10)
    print(f"Underperforming Bet Result: {res2}")
    assert res2["challenge_issued"] == True
    assert res2["penalty_debt"] == 1

    print("\n--- Test 3: Multiple Failures & Peer Booting ---")
    res3 = bonfire.place_zero_dollar_bet("http://localhost:8002", predicted_pas=0.05)
    res4 = bonfire.place_zero_dollar_bet("http://localhost:8002", predicted_pas=0.01)
    print(f"3rd Strike Result (Booted): {res4}")
    assert res4["booted"] == True
    assert "http://localhost:8002" not in bonfire.peers
    print("SUCCESS: Peer booted cleanly after 3 underperformance strikes!")

    bonfire.close()

def test_economic_agent_protocol_linker():
    print("\n--- Test 4: Sovereign Economic Agent & Protocol Linker ---")
    linker = EconomicAgentLinker(timeout=0.5)
    
    # 1. Bittensor Subnet Forecast & Compute Task Dispatch
    bittensor_data = linker.fetch_bittensor_subnet_prediction(41)
    print(f"Bittensor Subnet Query: {bittensor_data}")
    
    tao_submit = linker.submit_bittensor_subnet_forecast(41, {"predicted_pas": 0.88, "confidence": 0.95})
    print(f"Bittensor TAO Forecast Submission: {tao_submit}")
    
    tao_compute = linker.request_bittensor_subnet_compute(41, {"task": "ADMR_Matrix_Reduction", "dim": 64})
    print(f"Bittensor Compute Request: {tao_compute}")

    # 2. Autonolas / Olas Agent Mech Dispatch & Registry
    olas_data = linker.fetch_autonolas_olas_mech()
    print(f"Autonolas/Olas Mech Query: {olas_data}")
    
    olas_dispatch = linker.dispatch_olas_mech_task("0xOlasMechEndpoint", tool="prediction-settlement-v1", prompt="Settle prediction market bet #402")
    print(f"Autonolas Olas Mech Task Dispatch: {olas_dispatch}")
    
    olas_reg = linker.register_olas_agent_service("gyroidic_reasoner_node", "http://localhost:8000")
    print(f"Autonolas Service Registration: {olas_reg}")

    # 3. Sovereign News Ingestion
    news = linker.fetch_sovereign_business_news()
    print(f"Sovereign Business News items parsed: {len(news)}")
    print("ALL AGENT PROTOCOL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_zero_dollar_bet_and_booting()
    test_economic_agent_protocol_linker()
    sys.exit(0)
