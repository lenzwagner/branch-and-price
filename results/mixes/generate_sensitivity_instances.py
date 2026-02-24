#!/usr/bin/env python3
"""
Generate instances for Patient Mix Sensitivity Study

This script creates instances for three severity mix configurations:
- Baseline: Default DRG distribution (4.8%, 27.8%, 67.4%)
- Φ_Neuro: High-Complexity mix (70%, 20%, 10%)
- Φ_Bias-Free: High-Turnover mix (10%, 20%, 70%)
"""

import sys
import os

# Add mixes directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mixes'))

from generate_instances import InstanceGenerator

def main():
    print("\n" + "=" * 80)
    print(" PATIENT MIX SENSITIVITY STUDY - INSTANCE GENERATION ".center(80, "="))
    print("=" * 80 + "\n")
    
    print("Please enter the configuration:\n")
    
    # Interactive configuration
    # Seeds
    seeds_input = input("Seeds (comma-separated, e.g. '42,43,44,45,46'): ").strip()
    seeds = [int(s.strip()) for s in seeds_input.split(',')]
    
    # PTTR variants
    pttr_input = input("PTTR variants (comma-separated, e.g. 'light,medium,heavy'): ").strip()
    if pttr_input:
        pttr_variants = [p.strip() for p in pttr_input.split(',')]
    else:
        pttr_variants = ['medium']
    
    # T variants
    T_input = input("Therapists T (comma-separated, e.g. '10' or '8,10,12'): ").strip()
    T_variants = [int(t.strip()) for t in T_input.split(',')]
    
    # D_focus variants
    D_input = input("Focus days D (comma-separated, e.g. '30' or '20,30,40'): ").strip()
    D_focus_variants = [int(d.strip()) for d in D_input.split(',')]
    
    # Severity mix configurations
    print("\nSeverity Mix Configuration:")
    print("  1 = Baseline only")
    print("  2 = Baseline + Neuro")
    print("  3 = Baseline + Bias-Free")
    print("  4 = All three (Baseline + Neuro + Bias-Free)")
    print("  5 = Neuro only")
    print("  6 = Bias-Free only")
    print("  7 = Neuro + Bias-Free (without Baseline)")
    
    mix_choice = input("\nChoose an option (1-7) [Default: 4]: ").strip()
    
    if mix_choice == '1':
        severity_mix_variants = [(None, None)]
    elif mix_choice == '2':
        severity_mix_variants = [(None, None), ('neuro', (0.7, 0.2, 0.1))]
    elif mix_choice == '3':
        severity_mix_variants = [(None, None), ('biasfree', (0.33, 0.34, 0.33))]
    elif mix_choice == '5':
        severity_mix_variants = [('neuro', (0.7, 0.2, 0.1))]
    elif mix_choice == '6':
        severity_mix_variants = [('biasfree', (0.33, 0.34, 0.33))]
    elif mix_choice == '7':
        severity_mix_variants = [('neuro', (0.7, 0.2, 0.1)), ('biasfree', (0.33, 0.34, 0.33))]
    else:  # Default: 4 or any other input
        severity_mix_variants = [
            (None, None),
            ('neuro', (0.7, 0.2, 0.1)),
            ('biasfree', (0.33, 0.34, 0.33)),
        ]
    
    print("\n" + "=" * 80)
    print("Configuration:")
    print(f"  Seeds: {seeds}")
    print(f"  PTTR: {pttr_variants}")
    print(f"  Therapists (T): {T_variants}")
    print(f"  Focus days (D): {D_focus_variants}")
    print(f"  Severity mixes: {len(severity_mix_variants)}")
    for name, mix in severity_mix_variants:
        if name is None:
            print(f"    - Baseline (default distribution)")
        else:
            print(f"    - {name.capitalize()}: {mix}")
    print(f"\nTotal instances to generate: {len(seeds) * len(pttr_variants) * len(T_variants) * len(D_focus_variants) * len(severity_mix_variants)}")
    print("\n" + "=" * 80 + "\n")
    
    # Initialize generator
    generator = InstanceGenerator(
        seeds=seeds,
        pttr_variants=pttr_variants,
        T_variants=T_variants,
        D_focus_variants=D_focus_variants,
        severity_mix_variants=severity_mix_variants
    )
    
    # Generate all instances
    print("Starting instance generation...")
    generator.generate_all_instances()
    
    # Export to Excel with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'instances/instances_severity_study_{timestamp}.xlsx'
    print(f"\n\nExporting instances to: {output_file}")
    generator.export_to_excel(output_file)
    
    print("\n" + "=" * 80)
    print(" GENERATION COMPLETE ".center(80, "="))
    print("=" * 80)
    print(f"\nGenerated {len(generator.instances)} instances")
    print(f"Output file: {output_file}")
    print("\nNext steps:")
    print(f"  1. Review instances: open {output_file}")
    print(f"  2. Run simulations: python3 mixes/loop_main_learning.py --scenarios {output_file}")
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
