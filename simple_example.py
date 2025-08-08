from pcb_inspector import PCBInspector


def main():

    inspector = PCBInspector()
    
    test_image = "/home/hung/Coding_thing/fanny_kit/dataset/connector_nguon/4d14c36e9dd5148b4dc415.jpg"
    
    
    result = inspector.inspect_pcb(test_image, source_id="example_camera")
    
    print(result)


if __name__ == "__main__":
    main()
