#!/bin/bash
# Setup script for Feature Purity CLIP project

set -e  # Exit on error

echo "=========================================="
echo "Feature Purity CLIP - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python version is >= 3.8
required_version="3.8"
if ! python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8+ is required"
    exit 1
fi

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✓ CUDA is available"
    python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')" 2>/dev/null || true
    python -c "import torch; print(f'  Number of GPUs: {torch.cuda.device_count()}')" 2>/dev/null || true
else
    echo "⚠ CUDA not available - GPU acceleration disabled"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed successfully"

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p checkpoints
mkdir -p data
mkdir -p results
mkdir -p figures

echo "✓ Directories created"

# Download models (optional)
echo ""
read -p "Download pretrained models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading pretrained models..."
    
    cd checkpoints
    
    echo "  Downloading CC3M CLIP..."
    wget -q --show-progress https://www.dropbox.com/s/5jsthdm85r2nfpz/cc3m_clip.pt -O cc3m_clip.pt || echo "  Failed to download CC3M CLIP"
    
    echo "  Downloading CC3M LaCLIP..."
    wget -q --show-progress https://www.dropbox.com/s/k2e1tgsfmo0afme/cc3m_laclip.pt -O cc3m_laclip.pt || echo "  Failed to download CC3M LaCLIP"
    
    cd ..
    echo "✓ Models downloaded to checkpoints/"
else
    echo "Skipping model download"
    echo "You can download models later from:"
    echo "  https://github.com/LijieeFan/LaCLIP"
fi

# Run minimal test
echo ""
echo "Running verification test..."
cat > test_setup.py << 'EOF'
import sys
try:
    import torch
    import torchvision
    import timm
    import open_clip
    import sklearn
    import pandas
    import numpy
    from src.models import CLIP_VITB16
    
    print("✓ All imports successful")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Torchvision: {torchvision.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    sys.exit(0)
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
EOF

if python test_setup.py; then
    rm test_setup.py
    echo ""
    echo "=========================================="
    echo "✓ Setup complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Read QUICKSTART.md for quick start guide"
    echo "2. Run experiments in experiments/ directory"
    echo "3. See README.md for full documentation"
    echo ""
    echo "Happy experimenting! 🚀"
else
    rm test_setup.py
    echo ""
    echo "Setup completed with errors. Please check the error messages above."
    exit 1
fi
