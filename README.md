# DigitalPhenotyping
Environment Setup

1. Installing Anaconda
  *https://www.anaconda.com/distribution/
  *choose Python 3.7 version
  
2. Go to anaconda, install python 3.6 version
  conda install python==3.6
  
3. Git clone Mask RCNN github page
  https://github.com/matterport/Mask_RCNN
  
4. Install dependecies (use requirements.txt from this gitpage, not the one cloned from Mask_rCNN)
  pip3 install -r requirements.txt
  
5. Download COCO weights
  https://github.com/matterport/Mask_RCNN/releases
  Choose Mask R-CNN 2.0, mask_rcnn_coco.h5
  
 6. Run the script recognition.py for viewing the trained result of spikes
  python recognition.py path/to/model path/to/data/directory/ path/to/result/directory/
  python recognition.py D:/maskRCNN/DigitalPhenotypingVIP/spike/model/spike.h5 D:/maskRCNN/DigitalPhenotypingVIP/spike/target/       D:/maskRCNN/DigitalPhenotypingVIP/spike/result/
**NOTE: THE DIRECTORY MAY BE DIFFERENT
