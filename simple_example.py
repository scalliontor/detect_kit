from pcb_inspector import PCBInspector
import cv2
import json






def main():
    inspector = PCBInspector()
    
    test_image = "/home/hung/Coding_thing/fanny_kit/training_images_20250809_173553/captured_images/manual_20250809_173619_191.jpg"
    
    result, annotated_image = inspector.inspect_pcb(test_image)
    


    if annotated_image is not None:
        output_file = "pcb_inspection_result.jpg"
        cv2.imwrite(output_file, annotated_image)
        print(f"\n💾 Saved result: {output_file}")
    
    return result, annotated_image


if __name__ == "__main__":
    main()
