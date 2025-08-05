#!/usr/bin/env python3
"""
Verification script to confirm visualization component has been removed
"""

print("🔍 Verifying visualization removal...")

# Test 1: Check that src imports work without visualization
try:
    import src
    print("✓ src package imports successfully")
    print(f"✓ Available modules: {src.__all__}")
    
    # Verify visualization is not in __all__
    if 'visualization' not in src.__all__:
        print("✓ visualization successfully removed from __all__")
    else:
        print("✗ visualization still in __all__")
        
except Exception as e:
    print(f"✗ Error importing src: {e}")

# Test 2: Check that individual modules work
try:
    from src import yolo, image_processing, file_io
    print("✓ All expected modules import successfully")
except Exception as e:
    print(f"✗ Error importing modules: {e}")

# Test 3: Verify visualization import fails
try:
    from src import visualization
    print("✗ visualization still accessible (should not be)")
except ImportError:
    print("✓ visualization correctly removed (ImportError as expected)")
except Exception as e:
    print(f"? Unexpected error testing visualization: {e}")

# Test 4: Check file system
import os
vis_path = os.path.join('src', 'visualization')
if not os.path.exists(vis_path):
    print("✓ visualization directory successfully removed from filesystem")
else:
    print("✗ visualization directory still exists on filesystem")

print("\n🎉 Visualization removal verification complete!")
