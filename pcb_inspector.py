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
        self.model_path = model_path or os.path.join(self.base_path, 'best.pt')
        self.template_path = template_path or os.path.join(self.base_path, 'golden_template_relative.json')
        
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
    
    def _get_error_code_for_component(self, class_name, position_info=None):
        """Get Vietnamese error code for component."""
        error_map = {
            'Jack 24V': 'Error-03',
            'Connector 4P': 'Error-04', 
            'Connector 3P': 'Error-07'
        }
        
        if class_name == 'M3x6':
            if position_info and 'label' in position_info:
                if 'M3x6 #2' in position_info['label']:  # Closer to anchor3 (left)
                    return 'Error-01'
                else:  # M3x6 #1 (right)
                    return 'Error-02'
            return 'Error-01'  # Default
        
        if class_name == 'Connector 2P':
            if position_info and 'anchor1_distance' in position_info:
                if position_info['anchor1_distance'] < position_info.get('threshold', 500):
                    return 'Error-05'  # Closer to anchor1 (van)
                else:
                    return 'Error-06'  # Further from anchor1 (điện cực)
            return 'Error-05'  # Default
        
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
            
            # Find anchors
            results, anchors = self._find_anchors_retry(image)
            if anchors is None:
                result = self._create_error_result(source_id, timestamp, "ANCHORS_NOT_FOUND", "CRITICAL")
                return result, image.copy()
            
            # Calculate transformation
            transformation = self._calculate_rotation_aware_transformation(anchors)
            if transformation is None:
                result = self._create_error_result(source_id, timestamp, "TRANSFORMATION_FAILED", "CRITICAL")
                return result, image.copy()
            
            # Detect components
            detected_components = self._detect_components(results[0])
            
            # Check for missing components
            errors.extend(self._check_missing_components(detected_components, transformation, anchors))
            
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
    
    def _find_anchors_retry(self, image, brightness_tweaks=[0, -30, 30, -50, 50]):
        """Find required anchors with brightness retry - 4-anchor approach."""
        required_anchors = ['anchor1', 'anchor 3', 'anchor4', 'fake']
        
        for i, beta in enumerate(brightness_tweaks):
            if beta == 0:
                adjusted_image = image.copy()
            else:
                adjusted_image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
            
            # Use simple Ultralytics prediction with default settings (no custom thresholds)
            results = self.model.predict(adjusted_image, verbose=False)
            result = results[0]
            
            # Look for all 4 required anchors
            found_anchors = {}
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                if class_name in required_anchors:
                    x1, y1, x2, y2 = box.xyxy[0]
                    center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                    found_anchors[class_name] = {'center': center, 'confidence': confidence}
            
            # Check if we have all 4 anchors
            if len(found_anchors) >= 4:
                return results, found_anchors
        
        return None, None
    
    def _calculate_rotation_aware_transformation(self, anchors):
        """Calculate rotation-aware transformation using relative positioning."""
        # Get template anchor vectors (relative to anchor1)
        template_anchors = self.template['anchor_system']['reference_anchors']
        template_anchor1 = np.array(template_anchors['anchor1'])  # [0, 0]
        template_anchor3 = np.array(template_anchors['anchor 3'])
        template_anchor4 = np.array(template_anchors['anchor4'])
        template_fake = np.array(template_anchors['fake'])
        
        # Get detected anchor vectors (relative to anchor1)
        detected_anchor1 = np.array(anchors['anchor1']['center'])
        detected_anchor3 = np.array(anchors['anchor 3']['center']) - detected_anchor1
        detected_anchor4 = np.array(anchors['anchor4']['center']) - detected_anchor1
        detected_fake = np.array(anchors['fake']['center']) - detected_anchor1
        
        # Calculate scale factor using anchor1 to anchor3 as primary reference
        template_ref_dist = np.linalg.norm(template_anchor3)
        detected_ref_dist = np.linalg.norm(detected_anchor3)
        scale_factor = detected_ref_dist / template_ref_dist if template_ref_dist > 0 else 1.0
        
        # Calculate rotation angle using anchor1 to anchor3 vector
        template_angle = np.arctan2(template_anchor3[1], template_anchor3[0])
        detected_angle = np.arctan2(detected_anchor3[1], detected_anchor3[0])
        rotation_angle = detected_angle - template_angle
        
        # Create transformation parameters
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        
        # Rotation + Scale matrix
        rotation_scale_matrix = scale_factor * np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        return {
            'rotation_scale_matrix': rotation_scale_matrix,
            'translation': detected_anchor1,
            'scale_factor': scale_factor,
            'rotation_angle': rotation_angle
        }
    
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

    def _check_missing_components(self, detected_components, transformation, anchors):
        """Check for missing components using rotation-aware positioning."""
        errors = []
        
        # Special handling for M3x6 components (need to match pairs correctly)
        m3x6_templates = []
        other_templates = []
        
        for comp_template in self.template['components_relative']:
            if comp_template['class_name'] == 'M3x6':
                m3x6_templates.append(comp_template)
            else:
                other_templates.append(comp_template)
        
        # Process M3x6 components with proper pairing logic
        if m3x6_templates:
            errors.extend(self._check_m3x6_components(m3x6_templates, detected_components, transformation, anchors))
        
        # Process other components normally
        for comp_template in other_templates:
            comp_type = comp_template['class_name']
            template_rel_vec = np.array(comp_template['relative_vector'])
            
            # Apply rotation and scale transformation
            transformed_rel_vec = transformation['rotation_scale_matrix'] @ template_rel_vec
            expected_pos = transformation['translation'] + transformed_rel_vec
            
            # Check if this component type was detected AND has available instances
            if comp_type in detected_components and len(detected_components[comp_type]) > 0:
                detected_positions = detected_components[comp_type].copy()
                
                # Find closest detected component to expected position
                min_distance = float('inf')
                closest_detected = None
                
                for detected_pos in detected_positions:
                    distance = np.linalg.norm(np.array(detected_pos) - expected_pos)
                    if distance < min_distance:
                        min_distance = distance
                        closest_detected = detected_pos
                
                # Component found - remove from list to avoid double matching
                if closest_detected is not None:
                    detected_components[comp_type].remove(closest_detected)
            else:
                # Component type not detected at all OR no more instances available
                position_info = {'expected_pos': expected_pos.astype(int).tolist()}
                if comp_type == 'Connector 2P':
                    anchor1_distance = np.linalg.norm(expected_pos - np.array(anchors['anchor1']['center']))
                    position_info['anchor1_distance'] = anchor1_distance
                    position_info['threshold'] = 500
                
                description_text = f"Missing {comp_type} (not detected)"
                
                errors.append({
                    "error": self._get_error_code_for_component(comp_type, position_info),
                    "component": comp_type, 
                    "position": expected_pos.astype(int).tolist(),
                    "description": description_text
                })
        
        return errors

    def _check_m3x6_components(self, m3x6_templates, detected_components, transformation, anchors):
        """Special handling for M3x6 components with correct labeling."""
        errors = []
        
        # Transform all M3x6 template positions
        transformed_templates = []
        for i, comp_template in enumerate(m3x6_templates):
            template_rel_vec = np.array(comp_template['relative_vector'])
            transformed_rel_vec = transformation['rotation_scale_matrix'] @ template_rel_vec
            expected_pos = transformation['translation'] + transformed_rel_vec
            
            # Determine label based on template position relative to anchor3
            anchor3_distance = np.linalg.norm(expected_pos - np.array(anchors['anchor 3']['center']))
            anchor1_distance = np.linalg.norm(expected_pos - np.array(anchors['anchor1']['center']))
            
            label = 'M3x6 #2' if anchor3_distance < anchor1_distance else 'M3x6 #1'
            
            transformed_templates.append({
                'expected_pos': expected_pos,
                'template_vector': template_rel_vec,
                'label': label,
                'template_index': i
            })
        
        # Get detected M3x6 positions
        detected_m3x6 = detected_components.get('M3x6', []).copy()
        
        # Match detected M3x6 to expected positions
        matched_templates = []
        
        for template in transformed_templates:
            expected_pos = template['expected_pos']
            closest_detected = None
            min_distance = float('inf')
            closest_index = -1
            
            # Find closest detected M3x6 to this template position
            for j, detected_pos in enumerate(detected_m3x6):
                distance = np.linalg.norm(np.array(detected_pos) - expected_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_detected = detected_pos
                    closest_index = j
            
            # If found a reasonable match (within 200px), consider it matched
            if closest_detected is not None and min_distance < 200:
                matched_templates.append(template)
                detected_m3x6.pop(closest_index)  # Remove from available list
            else:
                # This M3x6 position is missing
                position_info = {'label': template['label']}
                
                errors.append({
                    "error": self._get_error_code_for_component('M3x6', position_info),
                    "component": 'M3x6', 
                    "position": expected_pos.astype(int).tolist(),
                    "description": f"Missing {template['label']} (not detected)"
                })
        
        # Clear the M3x6 from detected_components since we handled them
        if 'M3x6' in detected_components:
            detected_components['M3x6'] = []
        
        return errors
    
    def _create_visual_output(self, image, anchors, transformation, detected_components, errors):
        """Create annotated image with visual inspection results."""
        result_img = image.copy()
        
        # Draw detected anchors (green circles)
        for anchor_name, anchor_data in anchors.items():
            pos = anchor_data['center']
            cv2.circle(result_img, tuple(pos), 15, (0, 255, 0), 3)
            cv2.putText(result_img, anchor_name, (pos[0]+20, pos[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw coordinate system vectors for visualization
        origin = tuple(transformation['translation'].astype(int))
        for anchor_name in ['anchor 3', 'anchor4', 'fake']:
            if anchor_name in anchors:
                anchor_end = tuple(anchors[anchor_name]['center'])
                color = (255, 0, 0) if anchor_name == 'anchor 3' else (0, 255, 255) if anchor_name == 'anchor4' else (255, 0, 255)
                cv2.arrowedLine(result_img, origin, anchor_end, color, 2)
        
        # Draw matched components (green circles)
        for comp_type, positions in detected_components.items():
            for pos in positions:
                cv2.circle(result_img, tuple(pos), 15, (0, 255, 0), 3)
                cv2.putText(result_img, f"✓{comp_type}", (pos[0]+20, pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw BIG missing component boxes
        for error in errors:
            if 'position' in error:
                pos = error['position']
                # HUGE missing boxes (100x100 pixels)
                cv2.rectangle(result_img, (pos[0]-100, pos[1]-100), (pos[0]+100, pos[1]+100), (0, 0, 255), 6)
                # Large text with background
                text = f"✗{error['component']}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                # Draw text background
                cv2.rectangle(result_img, (pos[0]-120, pos[1]-140), 
                             (pos[0]-120+text_size[0]+20, pos[1]-140+text_size[1]+20), (0, 0, 255), -1)
                cv2.putText(result_img, text, (pos[0]-110, pos[1]-110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
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
