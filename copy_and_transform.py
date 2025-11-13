#!/usr/bin/env python3
"""Build script for FractalStat package"""

import re
from pathlib import Path
import shutil

SOURCE_ENGINE = Path("../packages/com.twg.the-seed/seed/engine")
DEST_DIR = Path("fractalstat")

CORE_FILES = [
    "stat7_entity.py",
    "stat7_experiments.py",
    "stat7_rag_bridge.py",
    "exp04_fractal_scaling.py",
    "exp05_compression_expansion.py",
    "exp06_entanglement_detection.py",
    "exp07_luca_bootstrap.py",
    "exp08_rag_integration.py",
    "exp09_concurrency.py",
    "bob_stress_test.py",
]

def transform_imports(content: str) -> str:
    content = re.sub(r'from seed\.engine\.', 'from fractalstat.', content)
    content = re.sub(r'import seed\.engine\.', 'import fractalstat.', content)
    return content

def main():
    print("Building FractalStat package...")
    DEST_DIR.mkdir(exist_ok=True)
    
    for file in CORE_FILES:
        src = SOURCE_ENGINE / file
        dst = DEST_DIR / file
        
        with open(src, 'r') as f:
            content = transform_imports(f.read())
        
        with open(dst, 'w') as f:
            f.write(content)
        
        print(f"✓ {file}")
    
    print("\n✓ Package built successfully!")

if __name__ == "__main__":
    main()
