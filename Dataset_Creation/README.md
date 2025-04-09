# Dataset Creation Process
This is the first step in the HI4Lines_Insp application: The Dataset Creation Process


## Run *python3 og2yolo.py --det_size <det_size> --class_size <class_size>*
*og2yolo.py* handles the following:

* parses the original IDID dataset
* creates two custom datasets: one for insulator detection (images and yolo labels) and one for health classification (binary) - each dataset has a custom resolution (default: 1280 for detection and 224 for classification)
* splits train/val/test at 64%/16%/20% of the total dataset
