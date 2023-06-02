# Application Guide

*Video Tutorial*

1. Run the application using "python3 app.py"

*Arguements to parse*
* -ip: path to image folder
* -rcp1: path to resnet weights for 3 against non-3 scores
* -rcp2: path to resnet weights for 0 against 1 against 2 scores
* -hcp: path to hrnet weights
* -v: viewing dimension of application on monitor/screen (Default: [1000, 1000])
* -d: device to use (Default: 'cpu')

2. Keyboard Keys

*Functions*
* b: Display area to be cropped
* c: Predict mSASSS scores
* m: Display and edit predicted mSASSS scores
* o: Overwrite and show original keypoints predictions
* p: Show latest keypoints prediction
* q: Quit application
* r: Hide keypoints prediction
* s: Save ammended keypoints prediction (COCO format and Image)
* v: Visibility option (Hide invisible keypoints - Not to be classified)
