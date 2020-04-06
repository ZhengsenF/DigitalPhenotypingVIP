# DigitalPhenotyping
Environment Setup

1. Installing Anaconda <br />
  https://www.anaconda.com/distribution/<br />
  choose Python 3.7 version<br />
  
2. Go to Anaconda prompt, install python 3.6 version<br />
```
  conda install python=3.6
```
3. Git clone Mask RCNN github page<br />
   (if you do not have Git installed, please do ```conda install git``` first)<br />
   Navigate to an empty directory and git clone the repository from Matterport, Inc<br />
```
  git clone https://github.com/matterport/Mask_RCNN
```
4. Git clone this repository<br />
   Navigate to another empty directory and git clone this repository
   
```
  https://github.com/ZhengsenF/DigitalPhenotypingVIP.git
```
  
5. Install dependecies (use requirements.txt from this gitpage, not the one cloned from Mask_rCNN)<br />
Navigate into DigitaPhenotypingVIP and then
```
  pip install -r requirements.txt
```
6. Download spike weight<br />
  Download trained weight from https://www.dropbox.com/s/4u34vk1fpgvkwi5/spike.h5?dl=0<br />
  
7. Run the script recognition.py for viewing the trained result of spikes<br />
    Copy recognition.py to the directory of MaskRCNN and navigate to that directory<br />
    Then type the following command<br />
 ```
  python recognition.py path/to/weight path/to/data/directory/ path/to/result/directory/
 ```
    For example
 ```
  python recognition.py D:/maskRCNN/DigitalPhenotypingVIP/spike/model/spike.h5 D:/maskRCNN/DigitalPhenotypingVIP/spike/target/       D:/maskRCNN/DigitalPhenotypingVIP/spike/result/
  ```
**NOTE: THE DIRECTORY MAY BE DIFFERENT<br />
