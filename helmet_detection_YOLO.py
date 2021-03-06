# -*- coding: utf-8 -*-
"""trafic_violation_helmet_detaction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lWQLx7x5j1IhzbTKsl4nlHuBlLyZjhbp
"""

# Check if NVIDIA GPU is enabled
!nvidia-smi

from google.colab import drive
drive.mount('/content/gdrive')
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive

"""**Cloning the darknet**  """

!git clone https://github.com/AlexeyAB/darknet

"""Compiling darknet-nvidia GPU"""

# Commented out IPython magic to ensure Python compatibility.
# change makefile to have GPU and OPENCV enabled
# %cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!make

"""**Configure yolov3**"""

!cp cfg/yolov3.cfg cfg/yolov3_training.cfg

"""We are using 5 classes <br>
1) With helmet <br>
2) Without helmet

So, for 5 class we have to change few configuration.<br>
Need to change -
*   Classes to 5
*   Filter to (n+5)*3 , n=classes so filter = 30
*  Max_baches to approx n*2000 , n->classs





"""

!sed -i 's/batch=1/batch=64/' cfg/yolov3_training.cfg
!sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov3_training.cfg
!sed -i 's/max_batches = 500200/max_batches = 10000/' cfg/yolov3_training.cfg
!sed -i '610 s@classes=80@classes=5@' cfg/yolov3_training.cfg
!sed -i '696 s@classes=80@classes=5@' cfg/yolov3_training.cfg
!sed -i '783 s@classes=80@classes=5@' cfg/yolov3_training.cfg
!sed -i '603 s@filters=255@filters=30@' cfg/yolov3_training.cfg
!sed -i '689 s@filters=255@filters=30@' cfg/yolov3_training.cfg
!sed -i '776 s@filters=255@filters=30@' cfg/yolov3_training.cfg

"""**Creating .name & .data files**"""

!echo -e 'with_helmet\nwithout_helmet\nperson\nbike\nlicense_plate' > data/obj.names
!echo -e 'classes= 5\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = /mydrive/yolov3' > data/obj.data

"""Saving yolov3_training.cfg and obj.names to drive for later use"""

!cp cfg/yolov3_training.cfg /mydrive/yolov3/yolov3_testing.cfg
!cp data/obj.names /mydrive/yolov3/classes.txt

"""Trained weight of convolutional layer """

!wget https://pjreddie.com/media/files/darknet53.conv.74

"""**Unziping image folder**"""

!mkdir data/obj
!unzip /mydrive/yolov3/finalDataset.zip -d data/obj

"""Generating file (train.txt) containing all images name """

import glob
images_list = glob.glob("data/obj/finalDataset/*.png")
with open("data/train.txt", "w") as f:
    f.write("\n".join(images_list))   
#print(images_list)

"""Training Model..."""

# !./darknet detector train data/obj.data cfg/yolov3_training.cfg darknet53.conv.74 -dont_show
!./darknet detector train data/obj.data cfg/yolov3_training.cfg /mydrive/yolov3/yolov3_training_last.weights -dont_show