# detect_kit

## Enhanced PCB Inspector Module

Advanced PCB inspection### Visual Output

- **🔴 Red**: Missing components (40x40 pixel boxes with identifiers)
- **📊 Clean Display**: Shows only errors, no clutter from detected components
- **🏷️ Text Labels**: Component identifiers for precise tracking (m3x6_1, connector2p_2, etc.)smart error detection and structured output.

### Quick Start

**Basic Usage:**
```python
from pcb_inspector import PCBInspector

# Initialize inspector
inspector = PCBInspector()

# Inspect an image - returns dict and annotated image
result, annotated_image = inspector.inspect_pcb("path/to/pcb_image.jpg")

# Check for errors
if result.get('errors'):
    for error in result['errors']:
        print(f"Missing: {error['component']} ({error['identifier']})")
```

### Testing

Run the simple example:
```bash
cd detect_kit
python simple_example.py
```

### Enhanced JSON Output

**Returns:**
```json
{
  "source_id": "camera_frame",
  "timestamp": 1723334567.123,
  "errors": [
    {
      "error": "Error-06-m3x6_2",
      "component": "M3x6",
      "identifier": "m3x6_2",
      "position": [234, 567],
      "description": "Missing m3x6_2"
    }
  ]
}
```

### Features

- **🎯 Smart Error Detection**: Intelligent component identification with distance-based analysis
- **🎮 3-Anchor System**: Robust positioning using anchor1, anchor 3, fake
- **🚀 GPU Acceleration**: Automatic CUDA detection and optimization
- **� Structured Output**: Clean JSON format with component identifiers
- **🔧 Configurable**: Modular design with flexible configuration options

### Error Codes

| Code | Component | Description |
|------|-----------|-------------|
| Error-06-m3x6_1 | M3x6 #1 | Missing M3x6 further from anchor3 |
| Error-06-m3x6_2 | M3x6 #2 | Missing M3x6 closer to anchor3 |
| Error-03-jack24v | Jack 24V | Missing power jack |
| Error-04-connector4p | Connector 4P | Missing 4-pin connector |
| Error-05-connector2p_1 | Connector 2P #1 | Missing 2-pin connector (close to anchor1) |
| Error-05-connector2p_2 | Connector 2P #2 | Missing 2-pin connector (far from anchor1) |
| Error-07-connector3p | Connector 3P | Missing 3-pin connector |

### Visual Output

- **🟢 Green**: Successfully detected components
- **🔴 Red**: Missing components (40x40 pixel boxes)
- **� Cyan**: Anchor points and coordinate system
- **� Text Labels**: Component identifiers for precise tracking

### Testing

Run the simple example:
```bash
cd detect_kit
python simple_example.py
```

**Expected Output:**
```
🚀 Simple PCB Inspector Test
========================================
🔧 Testing PCB Inspector

📸 Inspecting: manual_20250809_173621_794.jpg

📊 Results:
  Errors found: 2
  🚨 Error-06-m3x6_2: Missing M3x6 (m3x6_2)
  🚨 Error-06-m3x6_1: Missing M3x6 (m3x6_1)

💾 Saved result: pcb_inspection_result.jpg

✅ Test completed successfully!
```

