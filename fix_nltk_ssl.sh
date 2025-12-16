#!/bin/bash
# Script to fix NLTK SSL certificate issues on macOS

echo "Fixing NLTK SSL certificate issues..."
echo ""

# Try to find and run the Install Certificates script
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
CERT_SCRIPT="/Applications/Python ${PYTHON_VERSION}/Install Certificates.command"

if [ -f "$CERT_SCRIPT" ]; then
    echo "Found certificate installer: $CERT_SCRIPT"
    echo "Running it now..."
    "$CERT_SCRIPT"
    echo ""
    echo "Certificate installation complete!"
else
    echo "Could not find automatic certificate installer."
    echo "Trying alternative methods..."
    echo ""
    
    # Alternative: Try to find Python installation
    PYTHON_PATH=$(which python3)
    echo "Python found at: $PYTHON_PATH"
    echo ""
    echo "Please run one of these commands manually:"
    echo "  /Applications/Python\\ $(python3 --version | awk '{print $2}' | cut -d. -f1,2)/Install\\ Certificates.command"
    echo ""
    echo "Or download NLTK resources manually after fixing SSL:"
    echo "  python3 -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger maxent_ne_chunker words"
fi

echo ""
echo "After fixing certificates, download NLTK resources with:"
echo "  python3 -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger maxent_ne_chunker words"





