"""
Script to reinstall imageProcessingUtils package with model files included.
Run this script in your imageProcessingUtils repo directory.
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print("Output:", result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print("Error:", e.stderr.strip() if e.stderr else str(e))
        return False

def main():
    print("=== imageProcessingUtils Package Reinstallation ===")
    print("This script will reinstall the package with model files included.")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Run this script in the imageProcessingUtils repo directory.")
        return False
    
    # Check if models exist
    models_dir = Path("src/imageProcessingUtils/yolo/models")
    if not models_dir.exists():
        print(f"❌ Error: Models directory not found at {models_dir}")
        return False
    
    model_files = list(models_dir.glob("*.pt"))
    print(f"📦 Found {len(model_files)} model files to include:")
    for model_file in model_files:
        size_mb = model_file.stat().st_size / (1024*1024)
        print(f"  - {model_file.name} ({size_mb:.1f} MB)")
    
    if not model_files:
        print("⚠️  Warning: No .pt model files found!")
    
    # Step 1: Uninstall existing package
    print(f"\n🗑️  Uninstalling existing imageProcessingUtils package...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "imageProcessingUtils", "-y"], 
                  capture_output=True)
    
    # Step 2: Clean build artifacts
    import shutil
    for clean_dir in ["build", "dist", "src/imageProcessingUtils.egg-info"]:
        if Path(clean_dir).exists():
            print(f"🧹 Cleaning {clean_dir}")
            shutil.rmtree(clean_dir)
    
    # Step 3: Build package
    if not run_command([sys.executable, "-m", "build"], "Building package"):
        return False
    
    # Step 4: Install package in development mode
    if not run_command([sys.executable, "-m", "pip", "install", "-e", "."], 
                      "Installing package in development mode"):
        return False
    
    print("\n🎉 Package installation completed!")
    print("\nNext steps:")
    print("1. Run the diagnostic script: python diagnose_yolo_models.py")
    print("2. Test in your other repo to verify model loading works")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
