"""
Quick test: Can we import and use the clean evaluator?
"""

import sys
sys.path.insert(0, '/Users/jeonghwakim/codex_quant')

print("Testing clean evaluator import...")

try:
    from engines.mc.entry_evaluation_clean import get_clean_evaluator
    print("✅ Import successful!")
    
    evaluator = get_clean_evaluator()
    print(f"✅ Evaluator created: {evaluator}")
    print(f"✅ GPU evaluator available: {evaluator.gpu_evaluator is not None}")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting entry_evaluation.py import...")

try:
    from engines.mc.entry_evaluation import MonteCarloEntryEvaluationMixin
    print("✅ entry_evaluation.py import successful!")
    
    # Check if clean evaluator is available
    import engines.mc.entry_evaluation as ee_module
    if hasattr(ee_module, '_CLEAN_EVALUATOR_AVAILABLE'):
        print(f"✅ _CLEAN_EVALUATOR_AVAILABLE = {ee_module._CLEAN_EVALUATOR_AVAILABLE}")
    else:
        print("⚠️  _CLEAN_EVALUATOR_AVAILABLE not found")
    
except Exception as e:
    print(f"❌ entry_evaluation.py import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All imports successful!")
