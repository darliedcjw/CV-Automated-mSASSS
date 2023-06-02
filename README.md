# Folder Description (Automated mSASSS)
#**App**

Basic application that integrates a 2 staged deep learning model to monitor and score the progression of ankylosing spondylitis. The application allows users to ammend the keypoints predictions at stage 1 and subsequently send the correctly cropped images for classification at stage 2. Additionally, the application has the capability to save ammended datasets in the appropriate format (COCO and ImageFolder) for seamless finetuning of current weights in the event of data drift.


#**HRNet**

Contains the scripts to train HRNet for keypoints detection in the space between two vertebral bodies.

#**ResNet**

Contains the scripts to train ResNet152 for classification of mSASSS scores.
