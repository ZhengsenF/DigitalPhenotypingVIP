# DigitalPhenotyping
Environment Setup

1. Installing Anaconda <br />
  https://www.anaconda.com/distribution/<br />
  choose Python 3.7 version<br />
  
2. Go to anaconda, install python 3.6 version<br />
```
  conda install python==3.6
```
3. Git clone Mask RCNN github page<br />
  https://github.com/matterport/Mask_RCNN<br />
  
4. Install dependecies (use requirements.txt from this gitpage, not the one cloned from Mask_rCNN)<br />
```
  pip install -r requirements.txt
```
5. Download COCO weights<br />
  https://github.com/matterport/Mask_RCNN/releases<br />
  Choose Mask R-CNN 2.0, mask_rcnn_coco.h5<br />
  //maybe download spike weights from dropbox link provided. 
  
 6. Run the script recognition.py for viewing the trained result of spikes<br />
 ```
  python recognition.py path/to/model path/to/data/directory/ path/to/result/directory/
  python recognition.py D:/maskRCNN/DigitalPhenotypingVIP/spike/model/spike.h5 D:/maskRCNN/DigitalPhenotypingVIP/spike/target/       D:/maskRCNN/DigitalPhenotypingVIP/spike/result/
  ```
**NOTE: THE DIRECTORY MAY BE DIFFERENT<br />
