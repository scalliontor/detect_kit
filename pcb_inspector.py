"""
PCB Inspector Module - Enhanced PCB inspection with smart error detection
Clean implementation without Kalman tracking, focused on accuracy and simplicity
"""

import json
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from typing import Dict, List, Any, Tuple


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
        self.model_path = model_path or os.path.join(self.base_path, 'best(40).pt')
        self.template_path = template_path or os.path.join(self.base_path, 'golden_template_relative.json')
        
        # Load model and template
        self.model = None
        self.template = None
        self._load_resources()
        
        # Current anchors for distance calculations
        self.current_anchors = {}
    
    def _load_resources(self):
        """Load YOLO model and template."""
        try:
            self.model = YOLO(self.model_path)
            # Force GPU usage if available
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
            except ImportError:
                pass  # torch not available, use CPU
                
            # Load template - try 3-anchor first, then fallback
            template_3anchor_path = os.path.join(self.base_path, 'golden_template_3anchor_improved.json')
            if os.path.exists(template_3anchor_path):
                with open(template_3anchor_path, 'r') as f:
                    self.template = json.load(f)
            else:
                # Use default 3-anchor template if file doesn't exist
                self.template = self._create_default_3anchor_template()
                
        except Exception as e:
            raise RuntimeError(f"Failed to load resources: {e}")
    
    def _create_default_3anchor_template(self):
        """Create default 3-anchor template"""
        return {
            "anchor_system": {
                "origin": "anchor1",
                "reference_anchors": {
                    "anchor1": [0, 0],
                    "anchor 3": [-61, -395],
                    "fake": [51, 216]
                }
            },
            "components_relative": [
                {
                    "class_name": "Jack 24V",
                    "relative_vector": [-245, -218],
                    "confidence_threshold": 0.25,
                    "identifier": "jack24v"
                },
                {
                    "class_name": "Connector 4P", 
                    "relative_vector": [-324, 47],
                    "confidence_threshold": 0.25,
                    "identifier": "connector4p"
                },
                {
                    "class_name": "Connector 2P",
                    "relative_vector": [-323, 297],
                    "confidence_threshold": 0.25,
                    "identifier": "connector2p_1"
                },
                {
                    "class_name": "Connector 3P",
                    "relative_vector": [-318, 648],
                    "confidence_threshold": 0.25,
                    "identifier": "connector3p"
                },
                {
                    "class_name": "M3x6",
                    "relative_vector": [110, 755],
                    "confidence_threshold": 0.25,
                    "identifier": "m3x6_1"
                },
                {
                    "class_name": "Connector 2P",
                    "relative_vector": [-324, 469],
                    "confidence_threshold": 0.25,
                    "identifier": "connector2p_2"
                },
                {
                    "class_name": "M3x6",
                    "relative_vector": [108, -379],
                    "confidence_threshold": 0.25,
                    "identifier": "m3x6_2"
                }
            ],
            "expected_counts": {
                "Jack 24V": 1,
                "Connector 4P": 1,
                "Connector 2P": 2,
                "Connector 3P": 1,
                "M3x6": 2
            }
        }
    
    def _get_error_code_for_component(self, class_name, position_info=None):
        """Get Vietnamese error code for component."""
        error_map = {
            'Jack 24V': 'Error-03',
            'Connector 4P': 'Error-04', 
            'Connector 3P': 'Error-07',
            'Connector 2P': 'Error-05',
            'M3x6': 'Error-06'
        }
        
        # Enhanced error codes with identifier information
        if position_info and 'identifier' in position_info:
            identifier = position_info['identifier']
            base_error = error_map.get(class_name, 'Error-Unknown')
            return f"{base_error}-{identifier}"
        
        return error_map.get(class_name, f'Error-Unknown-{class_name}')
    
    def inspect_pcb(self, image_path: str, source_id: str = "camera_01") -> tuple:
        """
        Inspect PCB image and return structured results with visual output.
        
        Args:
            image_path: Path to PCB image
            source_id: Identifier for the image source
            
        Returns:
            Tuple containing (result_dict, annotated_image)
        """
        timestamp = time.time()
        errors = []
        annotated_image = None
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                result = self._create_error_result(source_id, timestamp, "IMAGE_LOAD_FAILED", "CRITICAL")
                return result, None
            
            # Run YOLO detection directly on original image
            yolo_result = self._run_yolo_detection(image)
            if yolo_result is None:
                result = self._create_error_result(source_id, timestamp, "DETECTION_FAILED", "CRITICAL")
                return result, image.copy()
            
            # Extract anchors using 3-anchor system
            anchors = self._extract_anchors(yolo_result)
            self.current_anchors = anchors
            
            if not anchors or 'anchor1' not in anchors or 'anchor 3' not in anchors:
                result = self._create_error_result(source_id, timestamp, "ANCHORS_NOT_FOUND", "CRITICAL")
                return result, image.copy()
            
            # Calculate 3-anchor transformation
            transformation = self._calculate_transformation_3anchor(anchors)
            if transformation is None:
                result = self._create_error_result(source_id, timestamp, "TRANSFORMATION_FAILED", "CRITICAL")
                return result, image.copy()
            
            # Extract detected components
            detected_components = self._extract_detected_components(yolo_result)
            
            # Analyze for missing components with smart logic
            errors.extend(self._analyze_missing_components(detected_components, transformation))
            
            # Create annotated image
            annotated_image = self._create_visual_output(image, anchors, transformation, detected_components, errors)
            
        except Exception as e:
            result = self._create_error_result(source_id, timestamp, f"INSPECTION_ERROR: {str(e)}", "CRITICAL")
            return result, image.copy() if 'image' in locals() else None
        
        result = {
            "source_id": source_id,
            "timestamp": timestamp,
            "errors": errors
        }
        
        return result, annotated_image
    
    def _run_yolo_detection(self, image):
        """Run YOLO detection on image"""
        try:
            results = self.model.predict(image, verbose=False)
            return results[0]  # Return first result
        except Exception as e:
            return None
    
    def _extract_anchors(self, yolo_result):
        """Extract anchor positions from YOLO results (3-anchor system)"""
        anchors = {}
        
        if yolo_result is None or yolo_result.boxes is None:
            return anchors
        
        for box in yolo_result.boxes:
            class_name = self.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            
            # Only look for 3 anchors: anchor1, anchor 3, fake
            if class_name in ['anchor1', 'anchor 3', 'fake']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                
                anchors[class_name] = {
                    'center': center,
                    'confidence': confidence
                }
        
        return anchors
    
    def _calculate_transformation_3anchor(self, anchors):
        """Calculate transformation matrix from 3 detected anchors"""
        # Check if we have minimum required anchors
        if 'anchor1' not in anchors or 'anchor 3' not in anchors:
            return None
        
        # Reference positions from template
        ref_anchor1 = np.array(self.template['anchor_system']['reference_anchors']['anchor1'])
        ref_anchor3 = np.array(self.template['anchor_system']['reference_anchors']['anchor 3'])
        
        # Detected positions
        det_anchor1 = np.array(anchors['anchor1']['center'])
        det_anchor3 = np.array(anchors['anchor 3']['center'])
        
        # Calculate vectors
        ref_vector = ref_anchor3 - ref_anchor1
        det_vector = det_anchor3 - det_anchor1
        
        # Calculate scale
        ref_length = np.linalg.norm(ref_vector)
        det_length = np.linalg.norm(det_vector)
        
        if ref_length == 0 or det_length == 0:
            return None
            
        scale = det_length / ref_length
        
        # Calculate rotation angle
        ref_angle = np.arctan2(ref_vector[1], ref_vector[0])
        det_angle = np.arctan2(det_vector[1], det_vector[0])
        rotation_angle = det_angle - ref_angle
        
        # Create rotation and scale matrix
        cos_r = np.cos(rotation_angle)
        sin_r = np.sin(rotation_angle)
        rotation_scale_matrix = scale * np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        # Translation vector
        translation = det_anchor1
        
        # Optional: Validate with fake anchor if available
        transformation_quality = 1.0
        if 'fake' in anchors:
            ref_fake = np.array(self.template['anchor_system']['reference_anchors']['fake'])
            det_fake = np.array(anchors['fake']['center'])
            
            # Transform reference fake position and compare
            transformed_fake = translation + rotation_scale_matrix @ ref_fake
            fake_error = np.linalg.norm(transformed_fake - det_fake)
            
            # Quality metric based on fake anchor error (lower is better)
            transformation_quality = max(0.0, 1.0 - fake_error / 100.0)  # 100px tolerance
        
        return {
            'rotation_scale_matrix': rotation_scale_matrix,
            'translation': translation,
            'scale': scale,
            'rotation_angle': rotation_angle,
            'quality': transformation_quality
        }
    
    def _extract_detected_components(self, yolo_result):
        """Extract detected component positions from YOLO results"""
        detected_components = {}
        
        if yolo_result is None or yolo_result.boxes is None:
            return detected_components
        
        # Collect all detections by type
        for box in yolo_result.boxes:
            class_name = self.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            
            # Skip anchors - only process components
            if class_name not in ['fake', 'anchor1', 'anchor2', 'anchor 3', 'anchor4']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                
                if class_name not in detected_components:
                    detected_components[class_name] = []
                    
                detected_components[class_name].append({
                    'center': center,
                    'confidence': confidence
                })
        
        # Sort by confidence (highest first) for each component type
        for class_name in detected_components:
            detected_components[class_name].sort(key=lambda x: x['confidence'], reverse=True)
        
        return detected_components
    
    def _analyze_missing_components(self, detected_components, transformation):
        """Analyze for missing components with smart identification logic"""
        errors = []
        
        if transformation is None:
            return errors
        
        # Get anchor positions for distance calculations
        anchors = self._get_current_anchor_positions()
        
        # Check each component type based on template expectations
        template_expectations = {
            'M3x6': {'count': 2, 'identifiers': ['m3x6_2', 'm3x6_1']},  # m3x6_2 closer to anchor3
            'Connector 2P': {'count': 2, 'identifiers': ['connector2p_1', 'connector2p_2']},  # connector2p_1 closer to anchor1
            'Connector 4P': {'count': 1, 'identifiers': ['connector4p']},
            'Connector 3P': {'count': 1, 'identifiers': ['connector3p']},
            'Jack 24V': {'count': 1, 'identifiers': ['jack24v']}
        }
        
        for comp_type, expectation in template_expectations.items():
            detected_instances = detected_components.get(comp_type, [])
            expected_count = expectation['count']
            
            if len(detected_instances) < expected_count:
                # Find which specific instances are missing
                missing_instances = self._identify_missing_instances(
                    comp_type, detected_instances, expected_count, transformation, anchors
                )
                errors.extend(missing_instances)
        
        return errors
    
    def _get_current_anchor_positions(self):
        """Get current anchor positions if available"""
        anchors = {}
        
        # Use stored current anchors if available
        if hasattr(self, 'current_anchors') and self.current_anchors:
            for anchor_name, anchor_data in self.current_anchors.items():
                anchors[anchor_name] = anchor_data['center']
        
        return anchors
    
    def _identify_missing_instances(self, comp_type, detected_instances, expected_count, transformation, anchors):
        """Identify which specific instances of a component type are missing"""
        errors = []
        
        # Get all template positions for this component type
        template_positions = []
        for comp_template in self.template['components_relative']:
            if comp_template['class_name'] == comp_type:
                template_rel_vec = np.array(comp_template['relative_vector'])
                transformed_rel_vec = transformation['rotation_scale_matrix'] @ template_rel_vec
                expected_pos = transformation['translation'] + transformed_rel_vec
                
                identifier = comp_template.get('identifier', comp_type)
                template_positions.append({
                    'position': expected_pos,
                    'identifier': identifier,
                    'template': comp_template
                })
        
        if expected_count == 1:
            # Single instance component
            if len(detected_instances) == 0:
                template_pos = template_positions[0] if template_positions else None
                if template_pos:
                    position_info = {'identifier': template_pos['identifier']}
                    errors.append({
                        "error": self._get_error_code_for_component(comp_type, position_info),
                        "component": comp_type,
                        "identifier": template_pos['identifier'],
                        "position": template_pos['position'].astype(int).tolist(),
                        "description": f"Missing {template_pos['identifier']}",
                    })
        
        elif expected_count == 2:
            # Multiple instance component - use smart matching
            if comp_type == 'M3x6':
                errors.extend(self._check_m3x6_instances(detected_instances, template_positions, transformation, anchors))
            elif comp_type == 'Connector 2P':
                errors.extend(self._check_connector2p_instances(detected_instances, template_positions, transformation, anchors))
        
        return errors
    
    def _check_m3x6_instances(self, detected_instances, template_positions, transformation, anchors):
        """Check M3x6 instances - m3x6_2 closer to anchor3, m3x6_1 further"""
        errors = []
        
        if not anchors.get('anchor 3'):
            # No anchor3 position, can't do smart identification
            return self._check_generic_multiple_instances(detected_instances, template_positions, 'M3x6')
        
        anchor3_pos = np.array(anchors['anchor 3'])
        
        # Sort template positions by distance to anchor3
        template_positions.sort(key=lambda tp: np.linalg.norm(tp['position'] - anchor3_pos))
        
        if len(detected_instances) == 0:
            # Both missing
            for template_pos in template_positions:
                position_info = {'identifier': template_pos['identifier']}
                errors.append({
                    "error": self._get_error_code_for_component("M3x6", position_info),
                    "component": "M3x6",
                    "identifier": template_pos['identifier'],
                    "position": template_pos['position'].astype(int).tolist(),
                    "description": f"Missing {template_pos['identifier']}",
                })
        
        elif len(detected_instances) == 1:
            # One missing - determine which one
            detected_pos = np.array(detected_instances[0]['center'])
            
            # Check which template position is closer to the detected instance
            distances = [np.linalg.norm(detected_pos - tp['position']) for tp in template_positions]
            closest_idx = np.argmin(distances)
            
            # The missing one is the other template position
            missing_idx = 1 - closest_idx
            missing_template = template_positions[missing_idx]
            
            position_info = {'identifier': missing_template['identifier']}
            errors.append({
                "error": self._get_error_code_for_component("M3x6", position_info),
                "component": "M3x6",
                "identifier": missing_template['identifier'],
                "position": missing_template['position'].astype(int).tolist(),
                "description": f"Missing {missing_template['identifier']}",
            })
        
        return errors
    
    def _check_connector2p_instances(self, detected_instances, template_positions, transformation, anchors):
        """Check Connector 2P instances - connector2p_1 closer to anchor1, connector2p_2 further"""
        errors = []
        
        if not anchors.get('anchor1'):
            # No anchor1 position, can't do smart identification
            return self._check_generic_multiple_instances(detected_instances, template_positions, 'Connector 2P')
        
        anchor1_pos = np.array(anchors['anchor1'])
        
        # Sort template positions by distance to anchor1
        template_positions.sort(key=lambda tp: np.linalg.norm(tp['position'] - anchor1_pos))
        
        if len(detected_instances) == 0:
            # Both missing
            for template_pos in template_positions:
                position_info = {'identifier': template_pos['identifier']}
                errors.append({
                    "error": self._get_error_code_for_component("Connector 2P", position_info),
                    "component": "Connector 2P",
                    "identifier": template_pos['identifier'],
                    "position": template_pos['position'].astype(int).tolist(),
                    "description": f"Missing {template_pos['identifier']}",
                })
        
        elif len(detected_instances) == 1:
            # One missing - determine which one
            detected_pos = np.array(detected_instances[0]['center'])
            
            # Check which template position is closer to the detected instance
            distances = [np.linalg.norm(detected_pos - tp['position']) for tp in template_positions]
            closest_idx = np.argmin(distances)
            
            # The missing one is the other template position
            missing_idx = 1 - closest_idx
            missing_template = template_positions[missing_idx]
            
            position_info = {'identifier': missing_template['identifier']}
            errors.append({
                "error": self._get_error_code_for_component("Connector 2P", position_info),
                "component": "Connector 2P",
                "identifier": missing_template['identifier'],
                "position": missing_template['position'].astype(int).tolist(),
                "description": f"Missing {missing_template['identifier']}",
            })
        
        return errors
    
    def _check_generic_multiple_instances(self, detected_instances, template_positions, comp_type):
        """Generic check for multiple instances when anchor positions are not available"""
        errors = []
        
        missing_count = len(template_positions) - len(detected_instances)
        if missing_count > 0:
            # Simple approach: assume missing instances are the template positions
            # that are furthest from detected instances
            for i, template_pos in enumerate(template_positions):
                if i >= len(detected_instances):  # Simple assumption
                    position_info = {'identifier': template_pos['identifier']}
                    errors.append({
                        "error": self._get_error_code_for_component(comp_type, position_info),
                        "component": comp_type,
                        "identifier": template_pos['identifier'],
                        "position": template_pos['position'].astype(int).tolist(),
                        "description": f"Missing {template_pos['identifier']}",
                    })
        
        return errors
    
    def _create_visual_output(self, image, anchors, transformation, detected_components, errors):
        """Create annotated image showing only errors (missing components)."""
        result_img = image.copy()
        
        # Only draw missing component boxes (40x40 pixels) - focus on errors only
        for error in errors:
            if 'position' in error:
                pos = error['position']
                color = (0, 0, 255)  # Red for missing components
                
                # Missing boxes (40x40 pixels) - smaller size
                box_size = 20  # Half-size from center (40x40 total)
                cv2.rectangle(result_img, (pos[0]-box_size, pos[1]-box_size), (pos[0]+box_size, pos[1]+box_size), color, 3)
                
                # Text with background - show identifier if available
                identifier = error.get('identifier', error['component'])
                text = f"✗{identifier}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]  # Smaller font
                
                # Draw text background (adjusted position)
                cv2.rectangle(result_img, (pos[0]-25, pos[1]-40), 
                             (pos[0]-25+text_size[0]+10, pos[1]-40+text_size[1]+10), color, -1)
                
                # Draw text (adjusted position)
                cv2.putText(result_img, text, (pos[0]-20, pos[1]-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_img
    
    def _create_error_result(self, source_id: str, timestamp: float, error_code: str, severity: str):
        """Create error result for system failures."""
        return {
            "source_id": source_id,
            "timestamp": timestamp,
            "errors": [{
                "error": error_code,
                "severity": severity
            }]
        }
    
    # Legacy compatibility methods - keeping the original method names
    def _find_anchors_retry(self, image, brightness_tweaks=[0, -30, 30, -50, 50]):
        """Legacy compatibility method - redirects to new YOLO detection"""
        yolo_result = self._run_yolo_detection(image)
        anchors = self._extract_anchors(yolo_result)
        return yolo_result, anchors
    
    def _calculate_rotation_aware_transformation(self, anchors):
        """Legacy compatibility method - redirects to 3-anchor transformation"""
        return self._calculate_transformation_3anchor(anchors)
    
    def _detect_components(self, result):
        """Legacy compatibility method - redirects to new component extraction"""
        return self._extract_detected_components(result)
    
    def _check_missing_components(self, detected_components, transformation, anchors):
        """Legacy compatibility method - redirects to new smart analysis"""
        return self._analyze_missing_components(detected_components, transformation)
