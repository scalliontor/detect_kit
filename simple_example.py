from pcb_inspector import PCBInspector
import cv2
import datetime


def main():
    inspector = PCBInspector()
    
    test_image = "/home/hung/Coding_thing/fanny_kit/dataset/m3phai/d39450250d9e84c0dd8f10.jpg"
    # Load image and run YOLO detection first
    image = cv2.imread(test_image)
    print(f"Image loaded: {image.shape}")
    
    # Run YOLO detection to see what's detected
    yolo_result = inspector.model.predict(image, verbose=False, conf=0.25)
    print(f"\n🔍 YOLO Detection Results:")
    
    if yolo_result and len(yolo_result) > 0 and yolo_result[0].boxes is not None:
        for box in yolo_result[0].boxes:
            class_name = inspector.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            print(f"  - {class_name}: {confidence:.3f} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    else:
        print("  - No detections found!")
    
    # Now run full inspection
    print(f"\n🔬 Full Inspection Results:")
    result, annotated_image = inspector.inspect_pcb(test_image)
    
    print(result)
    
    # Let's also manually check the template expectations
    print(f"\n📋 Template Expectations:")
    expected_counts = inspector.template.get('expected_counts', {})
    for comp_type, expected_count in expected_counts.items():
        actual_count = 0
        for box in yolo_result[0].boxes:
            class_name = inspector.model.names[int(box.cls[0])]
            if class_name == comp_type:
                actual_count += 1
        
        status = "✅ OK" if actual_count == expected_count else "❌ MISSING"
        print(f"  - {comp_type}: Expected {expected_count}, Found {actual_count} {status}")
    
    # Save result image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"inspection_result_{timestamp}.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"\n📸 Result saved to: {output_path}")




if __name__ == "__main__":
    main()
