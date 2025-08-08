#!/usr/bin/env python3
"""
Simple PCB Inspector Example
"""

import json
import os
from pcb_inspector import PCBInspector


def main():
    """Simple example of using PCB Inspector."""
    
    # Initialize the inspector
    inspector = PCBInspector()
    
    # Test image path
    test_image = "../dataset/m3phai/045684e7d95c5002094d9.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    # Run inspection
    result = inspector.inspect_pcb(test_image, source_id="example_camera")
    
    # Display results
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
