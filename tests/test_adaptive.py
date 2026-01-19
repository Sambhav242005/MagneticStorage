#!/usr/bin/env python3
"""
Test cases for Adaptive Threshold Clustering
Verifies that the threshold adjusts based on query patterns
"""

import sys
import threading
import time
from collections import deque

# Mock AdaptiveThreshold class (copy from neuro_savant.py for isolation)
class AdaptiveThreshold:
    MIN_THRESHOLD = 0.3
    MAX_THRESHOLD = 0.7
    INITIAL_THRESHOLD = 0.4
    PERCENTILE = 40
    
    def __init__(self, window_size=50):
        self.confidence_history = deque(maxlen=window_size)
        self.current_threshold = self.INITIAL_THRESHOLD
        self.lock = threading.Lock()
    
    def record_match(self, confidence: float, was_accepted: bool):
        with self.lock:
            self.confidence_history.append({
                'confidence': confidence,
                'accepted': was_accepted,
                'timestamp': time.time()
            })
            if len(self.confidence_history) >= 10 and len(self.confidence_history) % 10 == 0:
                self._update_threshold()
    
    def _update_threshold(self):
        if len(self.confidence_history) < 10:
            return
        confidences = sorted([h['confidence'] for h in self.confidence_history])
        idx = int(len(confidences) * (self.PERCENTILE / 100))
        new_threshold = confidences[idx]
        new_threshold = max(self.MIN_THRESHOLD, min(self.MAX_THRESHOLD, new_threshold))
        self.current_threshold = 0.7 * self.current_threshold + 0.3 * new_threshold
    
    def get_threshold(self) -> float:
        return self.current_threshold

def test_adaptive_threshold():
    """Test threshold adaptation logic"""
    print("=" * 60)
    print("Adaptive Threshold Tests")
    print("=" * 60)
    
    # 1. Test Initial State
    at = AdaptiveThreshold()
    initial = at.get_threshold()
    print(f"✅ Initial Threshold: {initial} (Expected: 0.4)")
    assert initial == 0.4
    
    # 2. Simulate HIGH CONFIDENCE patterns (Success Loop)
    # If users are asking very similar questions (conf ~ 0.8), threshold should rise
    # But it's dampened, so it won't jump instantly
    print("\n[Scenario 1] High Confidence Inputs (Typical matches ~0.8)")
    for i in range(20):
        at.record_match(confidence=0.8, was_accepted=True)
        if i % 5 == 0:
            print(f"   Sample {i+1}: Threshold = {at.get_threshold():.3f}")
            
    # After high confidence inputs, threshold should have increased
    final_high = at.get_threshold()
    print(f"   Result: {final_high:.3f}")
    if final_high > 0.4:
        print("✅ Threshold adapted UPWARDS (Correct)")
    else:
        print("❌ Threshold failed to adapt upwards")

    # 3. Simulate LOW CONFIDENCE patterns (Repulsion Loop)
    # Clear history to test adaptation independently (or wait for window to slide)
    at = AdaptiveThreshold(window_size=50) # Reset for clean test
    # Or keep same instance but push enough samples to flush history
    
    print("\n[Scenario 2] Low Confidence Inputs (Typical matches ~0.2)")
    # Start fresh with "new" environment
    at.current_threshold = 0.553 # Start from where high left off to verify drop
    
    for i in range(50): # Push 50 samples to fill window completely
        at.record_match(confidence=0.2, was_accepted=False)
        if i % 10 == 0:
            print(f"   Sample {i+1}: Threshold = {at.get_threshold():.3f}")
            
    final_low = at.get_threshold()
    print(f"   Result: {final_low:.3f}")
    
    # Ideally it should approach MIN_THRESHOLD (0.3)
    if final_low < 0.4:
        print("✅ Threshold adapted DOWNWARDS (Correct)")
    else:
        print(f"❌ Threshold failed to adapt downwards (Got {final_low})")
        
    # 4. Safety Bounds Test
    print("\n[Scenario 3] Safety Bounds Check")
    at_bounds = AdaptiveThreshold(window_size=50)
    
    # Try to force it super high (1.0)
    for _ in range(50):
        at_bounds.record_match(1.0, True)
    print(f"   Max Limit Check: {at_bounds.get_threshold():.3f} (Max allowed: 0.7)")
    assert at_bounds.get_threshold() <= 0.701, "Failed MAX bound"
    
    # Try to force it super low (0.0)
    for _ in range(50):
        at_bounds.record_match(0.0, False)
    print(f"   Min Limit Check: {at_bounds.get_threshold():.3f} (Min allowed: 0.3)")
    assert at_bounds.get_threshold() >= 0.299, "Failed MIN bound"
    
    print("✅ Safety bounds respected")

if __name__ == "__main__":
    test_adaptive_threshold()
