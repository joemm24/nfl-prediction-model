#!/bin/bash

# NFL Prediction Model - Setup Script
# This script sets up the development environment

set -e  # Exit on error

echo "=========================================="
echo "NFL Prediction Model - Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created (please update with your API keys)"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x setup.sh
chmod +x run_pipeline.py

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the full pipeline:"
echo "     python run_pipeline.py --full"
echo ""
echo "  3. Or run individual steps:"
echo "     python run_pipeline.py --fetch"
echo "     python run_pipeline.py --features"
echo "     python run_pipeline.py --train"
echo "     python run_pipeline.py --predict"
echo ""
echo "  4. Launch the dashboard:"
echo "     streamlit run src/dashboard.py"
echo ""
echo "  5. Start the API server:"
echo "     python src/api.py"
echo ""

