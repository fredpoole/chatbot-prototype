#!/usr/bin/env python3
"""
Helper script to download NLTK resources with SSL workaround.
Use this if you're having SSL certificate issues.
"""

import ssl
import nltk
import sys

# Disable SSL verification (workaround for certificate issues)
ssl._create_default_https_context = ssl._create_unverified_context

resources = [
    'punkt',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words',
    'punkt_tab',  # Optional but recommended
]

print("Downloading NLTK resources (SSL verification disabled)...")
print("=" * 60)

for resource in resources:
    print(f"\nDownloading {resource}...", end=' ', flush=True)
    try:
        nltk.download(resource, quiet=True)
        print("✓ Success")
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\n" + "=" * 60)
print("Done! If any downloads failed, you may need to:")
print("1. Fix SSL certificates: /Applications/Python\\ 3.12/Install\\ Certificates.command")
print("2. Then run: python3 -m nltk.downloader <resource_name>")




