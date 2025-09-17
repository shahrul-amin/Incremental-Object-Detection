# Configuration README

This README provides instructions on how to set up and customize the configuration files for continual object detection.

## Custom Annotation Method

To implement a custom annotation method by filtering out the class of annotation for only two objects, read the `README.md` in `data` folder.

## Dataset Configuration

The dataset configuration file is located inside the `datasets` folder. For the VOC dataset, the `voc.py` file is currently set up for only `car` and `person` classes. To adjust the path and classes for different training, follow these steps:

1. **Path Adjustment**:
    - Open the `voc.py` file in the `datasets` folder.
    - Adjust the dataset path to point to your desired Annotations and ImageSets location.

2. **Class Adjustment**:
    - Modify the class list in the `voc.py` file to include the objects you want to detect.
    - Ensure that the class names match the annotations in your dataset.
    - Set ```class_count = [0 for _ in range(2)]``` to desired class count 

