#!/usr/bin/env python3
"""
Simple test script to verify the main experiments are working.
"""

import sys
sys.path.insert(0, 'fractalstat')

def test_experiments():
    """Test the main experiments."""
    
    # Test EXP-01
    print('Testing EXP-01...')
    from fractalstat_experiments import EXP01_GeometricCollisionResistance
    exp01 = EXP01_GeometricCollisionResistance(sample_size=100)
    results01, success01 = exp01.run()
    print(f'EXP-01 Success: {success01}')
    
    # Test EXP-02  
    print('Testing EXP-02...')
    from fractalstat_experiments import EXP02_RetrievalEfficiency
    exp02 = EXP02_RetrievalEfficiency(query_count=100)
    results02, success02 = exp02.run()
    print(f'EXP-02 Success: {success02}')
    
    # Test EXP-08
    print('Testing EXP-08...')
    from exp08_self_organizing_memory import SelfOrganizingMemoryExperiment
    exp08 = SelfOrganizingMemoryExperiment(num_memories=100)
    results08 = exp08.run()
    print(f'EXP-08 Status: {results08.status}')
    
    all_passed = success01 and success02 and (results08.status == "PASS")
    print(f'All experiments passed: {all_passed}')
    
    return all_passed

if __name__ == "__main__":
    success = test_experiments()
    sys.exit(0 if success else 1)
