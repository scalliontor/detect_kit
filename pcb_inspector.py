"""
PCB Inspector Module - Simple PCB inspection with structured output
"""

import json
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from typing import Dict, List, Any


class PCBInspector:
    """PCB Inspector class for modular PCB component detection and analysis."""
    
    def __init__(self, model_path: str = None, template_path: str = None):
        """
        Initialize PCB Inspector.
        
        Args:
            model_path: Path to YOLO model file
            template_path: Path to golden template JSON file  
        """
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(self.base_path, 'best(2).pt')
        self.template_path = template_path or os.path.join(self.base_path, 'golden_template.json')
        
        # Load model and template
        self.model = None
        self.template = None
        self._load_resources()
    
    def _load_resources(self):
        """Load YOLO model and template."""
        try:
            self.model = YOLO(self.model_path)
            with open(self.template_path, 'r') as f:
                self.template = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load resources: {e}")
    
    def inspect_pcb(self, image_path: str, source_id: str = "camera_01") -> Dict[str, Any]:
        """
        Inspect PCB image and return structured results.
        
        Args:
            image_path: Path to PCB image
            source_id: Identifier for the image source
            
        Returns:
            Dict containing inspection results in specified format
        """
        timestamp = time.time()
        errors = []
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return self._create_error_result(source_id, timestamp, "IMAGE_LOAD_FAILED", "CRITICAL")
            
            # Find anchors
            results, anchors = self._find_anchors_retry(image)
            if anchors is None:
                return self._create_error_result(source_id, timestamp, "ANCHORS_NOT_FOUND", "CRITICAL")
            
            # Calculate transformation
            transformation_matrix = self._calculate_affine_transformation(anchors)
            if transformation_matrix is None:
                return self._create_error_result(source_id, timestamp, "TRANSFORMATION_FAILED", "CRITICAL")
            
            # Detect components
            detected_components = self._detect_components(results[0])
            
            # Analyze M3x6 positions
            m3x6_positions = [pos for pos in detected_components.get('M3x6', [])]
            m3x6_analysis = self._analyze_m3x6_with_anchor_proximity(
                m3x6_positions, anchors['anchor 3']['center'], transformation_matrix
            )
            
            # Check for missing components
            errors.extend(self._check_missing_components(detected_components, transformation_matrix))
            errors.extend(self._process_m3x6_analysis(m3x6_analysis))
            
        except Exception as e:
            return self._create_error_result(source_id, timestamp, f"INSPECTION_ERROR: {str(e)}", "CRITICAL")
        
        return {
            "source_id": source_id,
            "timestamp": timestamp,
            "errors": errors
        }
    
    def _find_anchors_retry(self, image, brightness_tweaks=[0, -30, 30, -50, 50]):
        """Find required anchors with brightness retry."""
        for i, beta in enumerate(brightness_tweaks):
            if beta == 0:
                adjusted_image = image.copy()
            else:
                adjusted_image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
            
            results = self.model.predict(adjusted_image, verbose=False)
            result = results[0]
            
            # Look for required anchors: fake, anchor1, anchor3
            found_anchors = {}
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                if class_name in ['fake', 'anchor1', 'anchor 3']:
                    x1, y1, x2, y2 = box.xyxy[0]
                    center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                    found_anchors[class_name] = {'center': center, 'confidence': confidence}
            
            # Check if we have at least 3 anchors
            if len(found_anchors) >= 3:
                return results, found_anchors
        
        return None, None
    
    def _calculate_affine_transformation(self, anchors):
        """Calculate affine transformation from template to image using 3 anchors."""
        # Get template anchor positions
        template_anchors = {}
        for anchor in self.template['fiducial_anchors']:
            if anchor['class_name'] == 'anchor4':
                template_anchors['fake'] = anchor['golden_center']
            else:
                template_anchors[anchor['class_name']] = anchor['golden_center']
        
        # Get detected anchor positions
        detected_anchors = {}
        for name, anchor_data in anchors.items():
            detected_anchors[name] = anchor_data['center']
        
        # Create point arrays for affine transformation
        template_points = []
        detected_points = []
        
        for anchor_name in ['anchor1', 'anchor 3', 'fake']:
            if anchor_name in template_anchors and anchor_name in detected_anchors:
                template_points.append(template_anchors[anchor_name])
                detected_points.append(detected_anchors[anchor_name])
        
        if len(template_points) < 3:
            return None
        
        template_points = np.array(template_points, dtype=np.float32)
        detected_points = np.array(detected_points, dtype=np.float32)
        
        # Calculate affine transformation
        transformation_matrix = cv2.getAffineTransform(template_points, detected_points)
        return transformation_matrix
    
    def _detect_components(self, result):
        """Detect and count components from YOLO results."""
        detected_components = {}
        
        for box in result.boxes:
            class_name = self.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            
            # Skip anchors
            if class_name not in ['fake', 'anchor1', 'anchor2', 'anchor 3', 'anchor4']:
                x1, y1, x2, y2 = box.xyxy[0]
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                if class_name not in detected_components:
                    detected_components[class_name] = []
                detected_components[class_name].append(center)
        
        return detected_components
    
    def _transform_template_position(self, template_pos, transformation_matrix):
        """Transform template position to image coordinates."""
        template_point = np.array([[template_pos[0], template_pos[1], 1]], dtype=np.float32)
        transformed_point = template_point @ transformation_matrix.T
        return [int(transformed_point[0][0]), int(transformed_point[0][1])]
    
    def _analyze_m3x6_with_anchor_proximity(self, detected_m3x6_positions, anchor3_pos, transformation_matrix):
        """Analyze M3x6 positions using anchor3 proximity."""
        if transformation_matrix is None:
            return {'missing_info': []}
        
        # Get M3x6 template positions
        m3x6_templates = []
        for comp in self.template['components_to_check']:
            if comp['class_name'] == 'M3x6':
                template_pos = comp['golden_center']
                transformed_pos = self._transform_template_position(template_pos, transformation_matrix)
                m3x6_templates.append({
                    'template_pos': template_pos,
                    'expected_pos': transformed_pos
                })
        
        missing_m3x6 = []
        
        if len(detected_m3x6_positions) == 0:
            # Both missing
            for i, m3x6 in enumerate(m3x6_templates):
                missing_m3x6.append({
                    'position': m3x6['expected_pos'],
                    'label': f"M3x6 #{i+1}",
                    'template_pos': m3x6['template_pos']
                })
        
        elif len(detected_m3x6_positions) == 1:
            # One missing - determine which one by distance to anchor3
            detected_pos = np.array(detected_m3x6_positions[0])
            anchor3_array = np.array(anchor3_pos)
            
            # Calculate distances to anchor3
            distance_to_anchor3 = np.linalg.norm(detected_pos - anchor3_array)
            template1_to_anchor3 = np.linalg.norm(np.array(m3x6_templates[0]['expected_pos']) - anchor3_array)
            template2_to_anchor3 = np.linalg.norm(np.array(m3x6_templates[1]['expected_pos']) - anchor3_array)
            
            # Determine which template position is closer to anchor3
            if template1_to_anchor3 < template2_to_anchor3:
                # Template #1 is closer to anchor3
                if distance_to_anchor3 < (template1_to_anchor3 + template2_to_anchor3) / 2:
                    # Detected M3x6 is M3x6 #1 (closer to anchor3), so #2 is missing
                    missing_m3x6.append({
                        'position': m3x6_templates[1]['expected_pos'],
                        'label': "M3x6 #2",
                        'template_pos': m3x6_templates[1]['template_pos']
                    })
                else:
                    # Detected M3x6 is M3x6 #2 (farther from anchor3), so #1 is missing
                    missing_m3x6.append({
                        'position': m3x6_templates[0]['expected_pos'],
                        'label': "M3x6 #1",
                        'template_pos': m3x6_templates[0]['template_pos']
                    })
            else:
                # Template #2 is closer to anchor3
                if distance_to_anchor3 < (template1_to_anchor3 + template2_to_anchor3) / 2:
                    # Detected M3x6 is M3x6 #2 (closer to anchor3), so #1 is missing
                    missing_m3x6.append({
                        'position': m3x6_templates[0]['expected_pos'],
                        'label': "M3x6 #1",
                        'template_pos': m3x6_templates[0]['template_pos']
                    })
                else:
                    # Detected M3x6 is M3x6 #1 (farther from anchor3), so #2 is missing
                    missing_m3x6.append({
                        'position': m3x6_templates[1]['expected_pos'],
                        'label': "M3x6 #2",
                        'template_pos': m3x6_templates[1]['template_pos']
                    })
        
        return {'missing_info': missing_m3x6}
    
    def _check_missing_components(self, detected_components, transformation_matrix):
        """Check for missing components and return error list."""
        errors = []
        
        # Expected counts from template
        expected_counts = {}
        for comp in self.template['components_to_check']:
            class_name = comp['class_name']
            expected_counts[class_name] = expected_counts.get(class_name, 0) + 1
        
        for class_name, expected_count in expected_counts.items():
            if class_name == 'M3x6':
                continue  # Handle M3x6 separately
            
            detected_count = len(detected_components.get(class_name, []))
            missing_count = expected_count - detected_count
            
            if missing_count > 0:
                # Find template position for missing component
                template_pos = None
                for comp in self.template['components_to_check']:
                    if comp['class_name'] == class_name:
                        template_pos = comp['golden_center']
                        break
                
                if template_pos and transformation_matrix is not None:
                    transformed_pos = self._transform_template_position(template_pos, transformation_matrix)
                    
                    # Create polygon around the expected position (rectangle)
                    box_size = 75  # Half the size of visual indicator
                    coordinates = [
                        [transformed_pos[0] - box_size, transformed_pos[1] - box_size],
                        [transformed_pos[0] + box_size, transformed_pos[1] - box_size],
                        [transformed_pos[0] + box_size, transformed_pos[1] + box_size],
                        [transformed_pos[0] - box_size, transformed_pos[1] + box_size]
                    ]
                    
                    errors.append({
                        "error_code": f"MISSING_{class_name.upper()}",
                        "severity": "WARNING",
                        "area_of_error": {
                            "type": "polygon",
                            "coordinates": coordinates
                        }
                    })
        
        return errors
    
    def _process_m3x6_analysis(self, m3x6_analysis):
        """Process M3x6 analysis results and return error list."""
        errors = []
        
        for missing_m3x6 in m3x6_analysis.get('missing_info', []):
            pos = missing_m3x6['position']
            label = missing_m3x6['label']
            
            # Create polygon around the expected position
            box_size = 75
            coordinates = [
                [pos[0] - box_size, pos[1] - box_size],
                [pos[0] + box_size, pos[1] - box_size],
                [pos[0] + box_size, pos[1] + box_size],
                [pos[0] - box_size, pos[1] + box_size]
            ]
            
            errors.append({
                "error_code": f"MISSING_{label.upper().replace(' ', '_')}",
                "severity": "WARNING",
                "area_of_error": {
                    "type": "polygon",
                    "coordinates": coordinates
                }
            })
        
        return errors
    
    def _create_error_result(self, source_id: str, timestamp: float, error_code: str, severity: str):
        """Create error result for system failures."""
        return {
            "source_id": source_id,
            "timestamp": timestamp,
            "errors": [{
                "error_code": error_code,
                "severity": severity,
                "area_of_error": {
                    "type": "rectangle",
                    "coordinates": [[0, 0], [0, 0], [0, 0], [0, 0]]
                }
            }]
        }
