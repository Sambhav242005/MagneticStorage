
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neuro_savant import NeuroSavant, Config

def test_infinite_flow():
    print("Testing Infinite Tool Integration...")
    
    # Init Agent
    config = Config(db_path="./neuro_savant_memory_test", model_name="granite3.1-moe:latest")
    agent = NeuroSavant(config)
    
    # Enable Infinite Mode
    if 'infinite' not in agent.tools:
        print("❌ Infinite tool not loaded!")
        return
        
    print(agent.tools['infinite'].execute("on"))
    
    # Trigger Chat
    response = agent.chat("Write Chapter 1 of a sci-fi novel about a lost key that unlocks a new dimension. Make it detailed.")
    
    if "Plan created" in response or "Story" in response or "Key" in response:
        print("\n✅ Verification passed: Output generated.")
        print(f"Output length: {len(response)}")
    else:
        print("\n❌ Verification failed: Unexpected output.")
        print(response)

if __name__ == "__main__":
    test_infinite_flow()
