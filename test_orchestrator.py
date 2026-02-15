#!/usr/bin/env python
"""Test the complete 7-phase training orchestrator."""

import sys
sys.path.insert(0, '.')

from pipeline.training_orchestrator import TrainingOrchestrator, TrainingConfig

print("\n" + "="*80)
print("Testing Complete 7-Phase Training Pipeline")
print("="*80 + "\n")

config = TrainingConfig(
    dataset_sources=[
        "https://kaggle.com/datasets/example1",
        "https://example.com/data.csv"
    ],
    problem_type="classification_multiclass",
    modalities=["image", "text", "tabular"],
    target_column="label"
)

orchestrator = TrainingOrchestrator(config)
results = orchestrator.run_pipeline()

print("\n" + "="*80)
print("Pipeline Results Summary")
print("="*80)
print(f"Status: {results['status']}")
print(f"Model ID: {results['model_id']}")
print(f"Total Duration: {results['total_duration_seconds']:.2f}s")
print(f"Device: {results['metadata']['device']}")
print("\nPhases Completed:")
for phase, data in results['phases'].items():
    phase_name = phase.name.replace('_', ' ').title()
    print(f"  [OK] {phase_name}: {data.get('duration_seconds', 'N/A'):.2f}s")
print("="*80 + "\n")
