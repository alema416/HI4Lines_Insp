# Dataset Creation Process
This is the first step in the HI4Lines_Insp application: The Dataset Creation Process

## First Step: download original IDID dataset

```
wget https://publicstorageaccnt.blob.core.windows.net/idid/Train_IDID_V1.2.zip
unzip Train_IDID_V1.2.zip -d ididv12
```
## Second Step: run the conversion code
 
```
python3 og2yolo.py --det_size <det_size> --class_size <class_size>
```
*og2yolo.py* handles the following:

* parses the original IDID dataset
* creates two custom datasets: one for insulator detection (images and yolo labels) and one for health classification (binary) - each dataset has a custom resolution (default: 1280 for detection and 224 for classification)
* splits train/val/test at 64%/16%/20% of the total dataset
