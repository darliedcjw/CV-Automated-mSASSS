import cv2
import os

from inference import SimpleHRNet

simplehrnet = SimpleHRNet(c=48, key=12, checkpoint_path='./logs/20221123_0906/checkpoint_best_acc.pth')

for image in os.listdir('./datasets/COCO/default_val'):
    image = cv2.imread(os.path.join('./datasets/COCO/default_val', image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    p, mv = simplehrnet.predict_single(image)

    for index, pair in enumerate(p[0]):
        if mv[0][index] > 0:
            image = cv2.circle(image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
        else:
            continue
    cv2.imshow('Dist', cv2.resize(image, (1000, 1000)))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        continue