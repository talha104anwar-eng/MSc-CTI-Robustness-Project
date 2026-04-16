"""
RUN ALL COMPONENTS - Master Script
MSc IT Dissertation - CTI Robustness Project

This script runs all four components sequentially:
1. Baseline Models
2. Adversarial Attacks
3. Defense Mechanisms
4. Explainability Analysis

Authors: Muhammad Sajid Iqbal, Aneela Shafique, Muhammad Arfan, Muhammad Talha Anwar
"""

import subprocess
import time
import sys

print("="*80)
print("MSc CTI ROBUSTNESS PROJECT - COMPLETE EXECUTION")
print("="*80)
print("\nThis will run all 4 components sequentially.")
print("Estimated total time: 20-30 minutes\n")

input("Press Enter to continue...")

components = [
    ("Component 1: Baseline Models", "component1_baseline_models.py"),
    ("Component 2: Adversarial Attacks", "component2_adversarial_attacks.py"),
    ("Component 3: Defense Mechanisms", "component3_defense_mechanisms.py"),
    ("Component 4: Explainability", "component4_explainability.py")
]

start_time = time.time()
results = []

for idx, (name, script) in enumerate(components, 1):
    print("\n" + "="*80)
    print(f"[{idx}/4] RUNNING: {name}")
    print("="*80)
    
    component_start = time.time()
    
    try:
        # Run component
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            text=True,
            check=True
        )
        
        component_time = time.time() - component_start
        results.append({
            'component': name,
            'status': 'SUCCESS',
            'time': component_time
        })
        
        print(f"\n✓ {name} completed in {component_time/60:.1f} minutes")
        
    except subprocess.CalledProcessError as e:
        component_time = time.time() - component_start
        results.append({
            'component': name,
            'status': 'FAILED',
            'time': component_time
        })
        
        print(f"\n✗ {name} FAILED after {component_time/60:.1f} minutes")
        print(f"Error: {e}")
        print("\nStopping execution...")
        break

total_time = time.time() - start_time

print("\n" + "="*80)
print("EXECUTION SUMMARY")
print("="*80)

for result in results:
    status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
    print(f"{status_symbol} {result['component']}: {result['status']} ({result['time']/60:.1f} min)")

print(f"\nTotal execution time: {total_time/60:.1f} minutes")

if all(r['status'] == 'SUCCESS' for r in results):
    print("\n" + "="*80)
    print("🎉 ALL COMPONENTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nResults available in:")
    print("  📁 results/     - CSV files and visualizations")
    print("  📁 models/      - Trained models (.pkl files)")
    print("\nNext steps:")
    print("  1. Review the results folder")
    print("  2. Check all PNG visualizations")
    print("  3. Upload to GitHub")
    print("  4. Send repository link to supervisor")
    print("="*80)
else:
    print("\n⚠️  Some components failed. Please check error messages above.")
