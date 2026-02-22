#!/usr/bin/env python3
"""
Script to regenerate figures with corrected MJ/m²/day unit labels
"""
import sys
sys.path.insert(0, '/DST481/solar-vehicle-ai/solar-vehicle-energy-forecasting')

try:
    from src.model_training import ModelTrainer
    import os
    
    # Initialize trainer 
    trainer = ModelTrainer()
    
    # Load data
    print("Loading data...")
    trainer.load_data()
    
    # Train models (this will generate all figures)
    print("Training models and generating figures...")
    trainer.train_all_models()
    
    print("\n" + "="*60)
    print("✓ Figures regenerated successfully with corrected labels!")
    print("="*60)
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
