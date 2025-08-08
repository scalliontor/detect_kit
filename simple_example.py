from pcb_inspector import PCBInspector
import cv2
import datetime


def main():
    inspector = PCBInspector()
    
    test_image = "/home/hung/Coding_thing/fanny_kit/d8bbffc6-35af-4ec9-a5a6-e7a486186cb8.jpeg"
    
    result, image = inspector.inspect_pcb(test_image)
    
    print(result)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"inspection_result_{timestamp}.jpg"
    cv2.imwrite(output_path, image)




if __name__ == "__main__":
    main()
