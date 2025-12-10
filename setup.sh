#!/bin/bash
# Setup script for RetNet State Extraction project
# Run this on your GPU machine after transferring files

set -e  # Exit on error

echo "=================================================="
echo "RetNet State Extraction - Setup Script"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "Found: $python_version"
echo ""

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Requirements installed"
echo ""

# Install FLA library
echo "Installing Flash Linear Attention library..."
pip install git+https://github.com/sustcsonglin/flash-linear-attention.git
echo "✓ FLA library installed"
echo ""

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'  ✓ Transformers {transformers.__version__}')"
python -c "import numpy; print(f'  ✓ NumPy {numpy.__version__}')"
python -c "import pandas; print(f'  ✓ Pandas {pandas.__version__}')"
python -c "import matplotlib; print(f'  ✓ Matplotlib {matplotlib.__version__}')"

# Check CUDA
echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA devices: {torch.cuda.device_count()}' if torch.cuda.is_available() else '  No CUDA devices found')"

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Run tests: python tests/test_state_shapes.py"
echo "  2. Or use Jupyter: jupyter notebook notebooks/01_test_state_extraction.ipynb"
echo ""
echo "See QUICKSTART.md for detailed usage instructions."

