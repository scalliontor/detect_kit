**Detect kit:**
```python
from pcb_inspector import PCBInspector

# Initialize inspector
inspector = PCBInspector()

# Inspect an image - returns dict and annotated image
result, annotated_image = inspector.inspect_pcb("path/to/pcb_image.jpg")

**Returns:**
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

**Error Codes**

| Code | Component | Description |
|------|-----------|-------------|
| Error-06-m3x6_1 | M3x6 #1 | Missing M3x6 further from anchor3 |
| Error-06-m3x6_2 | M3x6 #2 | Missing M3x6 closer to anchor3 |
| Error-03-jack24v | Jack 24V | Missing power jack |
| Error-04-connector4p | Connector 4P | Missing 4-pin connector |
| Error-05-connector2p_1 | Connector 2P #1 | Missing 2-pin connector (close to anchor1) |
| Error-05-connector2p_2 | Connector 2P #2 | Missing 2-pin connector (far from anchor1) |
| Error-07-connector3p | Connector 3P | Missing 3-pin connector |



