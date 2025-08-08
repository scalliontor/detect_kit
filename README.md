

### 1. Basic Usage

```python
from pcb_inspector import PCBInspector

# Initialize inspector
inspector = PCBInspector()

# Inspect an image - returns dict directly
result = inspector.inspect_pcb("path/to/pcb_image.jpg")
```
**Returns:**
```json
{
  "source_id": "camera_01",
  "timestamp": 1722345678.123,
  "errors": [
    {
      "error_code": "MISSING_M3X6_#1",
      "severity": "WARNING", 
      "confidence": 0.90,
      "area_of_error": {
        "type": "polygon",
        "coordinates": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
      }
    }
  ]
}
```

## Files

- `pcb_inspector.py` - Main inspector class
- `simple_example.py` - Basic example

The module automatically loads:
- Model: `best(2).pt` (local)
- Template: `golden_template.json` (local)
