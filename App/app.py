import os
from typing import List
import numpy as np
import tkinter as tk
import cv2
import json
import argparse
import torch
from torchvision import transforms
from utils import affine_transform, get_angle
from inference import SimpleHRNet, SimpleResNet152



def app(device, res_cpath_1, res_cpath_2):
    global resize_image, p, mv, index_store, width, height

    '''
    b: bounding box
    c: classify
    m: edit mSASSS scores
    o: original points
    p: plot points
    q: quit
    r: clear points
    s: save annotation
    v: visibility checkbox
    '''

    index_store = []
    confidence_store = []
    class_idx_store = []

    class_idx_1 = ['3', 'no 3']
    class_idx_2 = ['0', '1', '2']

    while(True):
        cv2.setMouseCallback(windowName, CallBackFunc)
        cv2.imshow(windowName, resize_image)
        
        key = cv2.waitKey(0)
        

        ########################################################################## Bounding Box ##########################################################################
        if key == ord('b'):
            print('\nDisplay bounding boxes!\n')
            avg_len = abs(p[3] - p[5]).mean()
            for index, pair in enumerate(p):
                if index in index_store:
                    continue
                else:
                    if index % 2 == 1:
                        pt1 = (int(pair[0] - avg_len), int(pair[1] - avg_len))
                        pt2 = (int(pair[0] + avg_len), int(pair[1] + avg_len))
                        resize_image = cv2.rectangle(resize_image, pt1=pt1, pt2=pt2, color=(255, 255, 255), thickness=1)
                    else:
                        continue


        ########################################################################## Classify ##########################################################################
        elif key == ord('c'):
            print('\nClassifying!\n')

            confidence_store.clear()
            class_idx_store.clear()
            p_copy = p.copy()
            m_reverse = cv2.getAffineTransform(src=dst, dst=src)
            for pair in p_copy:
                pair[:2] = affine_transform(pair, m_reverse, train=False)  

            # Crop
            assert 3 not in index_store and 5 not in index_store, 'Point 4 and Point 6 not visible for comptuation of y buffer!'
            avg_len = abs(p_copy[3] - p_copy[5]).mean()

            images_list = []

            for index, pair in enumerate(p_copy):
                if index % 2 == 1 and index not in index_store:
                    # Rotation
                    pt1, pt2 = p_copy[index - 1][:2], pair[:2]
                    rot_deg = get_angle(pt1=pt1, pt2=pt2)
            
                    # Affine transform for each of the outer points
                    m = cv2.getRotationMatrix2D(center=(width // 2, height // 2), angle=rot_deg, scale=1)
                    image_rot = cv2.warpAffine(src=image, M=m, dsize=(width, height))
                    x, y = affine_transform(pt2, m, train=False)
                    rot_width, rot_height = image_rot.shape[1], image_rot.shape[0]

                    # Clip
                    low_pt = (np.clip(int(x - avg_len), a_min=0, a_max=rot_width), np.clip(int(y - avg_len), a_min=0, a_max=rot_height))
                    high_pt = (np.clip(int(x + avg_len), a_min=0, a_max=rot_width), np.clip(int(y + avg_len), a_min=0, a_max=rot_height))

                    # Classification: 3
                    image_crop = image_rot[low_pt[1]:high_pt[1] + 1, low_pt[0]:high_pt[0] + 1, :]
                    image_crop = cv2.resize(image_crop, (224, 224))
                    images_list.append(transforms.ToTensor()(image_crop))
            
            images_no3_3 = torch.stack(images_list)

            simpleres_no3_3 = SimpleResNet152(num_class=2, checkpoint_path=res_cpath_1, resolution=(224, 224), device=torch.device(device))
            idx_no3_3, confidence_no3_3 = simpleres_no3_3.predict_batch(images_no3_3)
            idx_no3 = torch.tensor((range(len((idx_no3_3 == 1)))))

            idx_no3_3 = idx_no3_3.tolist()
            confidence_no3_3 = confidence_no3_3.tolist()

            crop_list = []

            for idx in idx_no3:
                upper = images_no3_3[idx, :, :112, :]
                lower = images_no3_3[idx, :, 112:, :]
                crop_list.append(upper)
                crop_list.append(lower)
            
            images_012 = torch.stack(crop_list)

            simpleres_012 = SimpleResNet152(num_class=3, checkpoint_path=res_cpath_2, resolution=(224, 224), device=torch.device(device))
            idx_012, confidence_012 = simpleres_012.predict_batch(images_012)

            idx_012 = idx_012.tolist()
            confidence_012 = confidence_012.tolist()


            for index, _ in enumerate(p_copy):
                if (index % 2 == 1) and (index not in index_store) and (idx_no3_3[0] == 0):
                    # 3 for Upper and Lower
                    confidence_store.append('{:.4f}'.format(confidence_no3_3[0]))
                    confidence_store.append('{:.4f}'.format(confidence_no3_3.pop(0)))
                    class_idx_store.append('{}'.format(class_idx_1[idx_no3_3[0]]))
                    class_idx_store.append('{}'.format(class_idx_1[idx_no3_3.pop(0)]))

                elif (index % 2 == 1) and (index not in index_store) and (idx_no3_3[0] == 1):
                    # Remove for Non-3
                    confidence_no3_3.pop(0)
                    idx_no3_3.pop(0)

                    # Upper
                    confidence_store.append('{:.4f}'.format(confidence_012.pop(0)))
                    class_idx_store.append('{}'.format(class_idx_2[idx_012.pop(0)]))

                    # Lower
                    confidence_store.append('{:.4f}'.format(confidence_012.pop(0)))
                    class_idx_store.append('{}'.format(class_idx_2[idx_012.pop(0)]))                    

                elif index % 2 ==1 and index in index_store:
                    confidence_store.append('Not Applicable')
                    confidence_store.append('Not Applicable')                    
                    class_idx_store.append('Not Applicable')
                    class_idx_store.append('Not Applicable')
                        
            zip_list = list(zip(class_idx_store, confidence_store))

            print('\nDone\n')


        ########################################################################## View and Save ##########################################################################
        elif key == ord('m'):

            if 'zip_list' not in locals():
                print('Please run classification first!')

            elif 'zip_list' in locals():
                window = tk.Tk()
                window.title('mSASSS Scores')
                window.minsize(width=200, height=300)
                window.grid_columnconfigure((0, 1, 2), weight=1)

                # Location
                tk.Label(master=window, text='Location').grid(row=0, column=0)
                tk.Label(master=window, text='Point 2 Upper').grid(row=1, column=0)
                tk.Label(master=window, text='Point 2 Lower').grid(row=2, column=0)
                tk.Label(master=window, text='Point 4 Upper').grid(row=3, column=0)
                tk.Label(master=window, text='Point 4 Lower').grid(row=4, column=0)
                tk.Label(master=window, text='Point 6 Upper').grid(row=5, column=0)
                tk.Label(master=window, text='Point 6 Lower').grid(row=6, column=0)
                tk.Label(master=window, text='Point 8 Upper').grid(row=7, column=0)
                tk.Label(master=window, text='Point 8 Lower').grid(row=8, column=0)
                tk.Label(master=window, text='Point 10 Upper').grid(row=9, column=0)
                tk.Label(master=window, text='Point 10 Lower').grid(row=10, column=0)
                tk.Label(master=window, text='Point 12 Upper').grid(row=11, column=0) 
                tk.Label(master=window, text='Point 12 Lower').grid(row=12, column=0)

                # Class
                tk.Label(master=window, text='Class').grid(row=0, column=1)
                
                entry_1 = tk.Entry(master=window)
                entry_1.insert(0, zip_list[0][0])
                entry_1.grid(row=1, column=1)

                entry_2 = tk.Entry(master=window)
                entry_2.insert(0, zip_list[1][0])
                entry_2.grid(row=2, column=1)

                entry_3 = tk.Entry(master=window)
                entry_3.insert(0, zip_list[2][0])
                entry_3.grid(row=3, column=1)

                entry_4 = tk.Entry(master=window)
                entry_4.insert(0, zip_list[3][0])
                entry_4.grid(row=4, column=1)

                entry_5 = tk.Entry(master=window)
                entry_5.insert(0, zip_list[4][0])
                entry_5.grid(row=5, column=1)

                entry_6 = tk.Entry(master=window)
                entry_6.insert(0, zip_list[5][0])
                entry_6.grid(row=6, column=1)

                entry_7 = tk.Entry(master=window)
                entry_7.insert(0, zip_list[6][0])
                entry_7.grid(row=7, column=1)

                entry_8 = tk.Entry(master=window)
                entry_8.insert(0, zip_list[7][0])
                entry_8.grid(row=8, column=1)

                entry_9 = tk.Entry(master=window)
                entry_9.insert(0, zip_list[8][0])
                entry_9.grid(row=9, column=1)

                entry_10 = tk.Entry(master=window)
                entry_10.insert(0, zip_list[9][0])
                entry_10.grid(row=10, column=1)

                entry_11 = tk.Entry(master=window)
                entry_11.insert(0, zip_list[10][0])
                entry_11.grid(row=11, column=1)

                entry_12 = tk.Entry(master=window)
                entry_12.insert(0, zip_list[11][0])
                entry_12.grid(row=12, column=1)

                # Confidence
                tk.Label(master=window, text='Confidence').grid(row=0, column=2)
                tk.Label(master=window, text=zip_list[0][1]).grid(row=1, column=2)
                tk.Label(master=window, text=zip_list[1][1]).grid(row=2, column=2)
                tk.Label(master=window, text=zip_list[2][1]).grid(row=3, column=2)
                tk.Label(master=window, text=zip_list[3][1]).grid(row=4, column=2)
                tk.Label(master=window, text=zip_list[4][1]).grid(row=5, column=2)
                tk.Label(master=window, text=zip_list[5][1]).grid(row=6, column=2)
                tk.Label(master=window, text=zip_list[6][1]).grid(row=7, column=2)
                tk.Label(master=window, text=zip_list[7][1]).grid(row=8, column=2)
                tk.Label(master=window, text=zip_list[8][1]).grid(row=9, column=2)
                tk.Label(master=window, text=zip_list[9][1]).grid(row=10, column=2)
                tk.Label(master=window, text=zip_list[10][1]).grid(row=11, column=2) 
                tk.Label(master=window, text=zip_list[11][1]).grid(row=12, column=2)

                # Saving Edited Class Files
                def record():
            
                    class_idx_store_new = []
                    class_idx_store_new.append(entry_1.get())
                    class_idx_store_new.append(entry_2.get())
                    class_idx_store_new.append(entry_3.get())
                    class_idx_store_new.append(entry_4.get())
                    class_idx_store_new.append(entry_5.get())
                    class_idx_store_new.append(entry_6.get())
                    class_idx_store_new.append(entry_7.get())
                    class_idx_store_new.append(entry_8.get())
                    class_idx_store_new.append(entry_9.get())
                    class_idx_store_new.append(entry_10.get())
                    class_idx_store_new.append(entry_11.get())
                    class_idx_store_new.append(entry_12.get())

                    if class_idx_store_new != class_idx_store:
                        
                        # Check Annotation Path Exist
                        if os.path.exists('Train_Case/datasets'):
                            pass
                        else:
                            os.makedirs('Train_Case/datasets/1/3')
                            os.makedirs('Train_Case/datasets/1/no_3')
                            os.makedirs('Train_Case/datasets/2/0')
                            os.makedirs('Train_Case/datasets/2/1')
                            os.makedirs('Train_Case/datasets/2/2')

                        # Create Count
                        last_count_3 = len(os.listdir('Train_Case/datasets/1/3'))
                        last_count_no3 = len(os.listdir('Train_Case/datasets/1/no_3'))
                        last_count_0 = len(os.listdir('Train_Case/datasets/2/0'))
                        last_count_1 = len(os.listdir('Train_Case/datasets/2/1'))
                        last_count_2 = len(os.listdir('Train_Case/datasets/2/2'))

                        new_count_3 = last_count_3 + 1
                        new_count_no3 = last_count_no3 + 1
                        new_count_0 = last_count_0 + 1
                        new_count_1 = last_count_1 + 1
                        new_count_2 = last_count_2 + 1

                        idx_changes = ((np.array(class_idx_store_new) == np.array(class_idx_store)) == False).nonzero()[0]
                        
                        # First Check for score 3
                        for index, idx in enumerate(idx_changes):
                            class_change = class_idx_store_new[idx]

                            if (class_change == '3') and (idx % 2 == 0):
                                assert idx_changes.shape[0] > 1, 'Should have 2 changes for mSASSS score 3'
                                assert idx_changes[index] + 1 == idx_changes[index + 1], 'Problem 1: Please ensure that both upper and lower border are of mSASSS score 3'
                                assert class_idx_store_new[idx_changes[index + 1]] == '3', 'Problem 2: Please ensure that both upper and lower border are of mSASSS score 3'

                            elif (class_change == '3') and (idx % 2 == 1):
                                assert idx_changes.shape[0] > 1, 'Should have 2 changes for mSASSS score 3'
                                assert idx_changes[index] - 1 == idx_changes[index - 1], 'Problem 1: Please ensure that both upper and lower border are of mSASSS score 3'
                                assert class_idx_store_new[idx_changes[index - 1]] == '3', 'Please ensure that both upper and lower border are of mSASSS score 3'
                            
                        # Save
                        for index, idx in enumerate(idx_changes):

                            class_change = class_idx_store_new[idx]

                            # 3
                            if (class_change == '3') and (idx % 2 == 0):        
                                image_save = images_no3_3[idx // 2]
                                cv2.imwrite('Train_Case/datasets/1/3/{}.png'.format(new_count_3), image_save.numpy().transpose(1, 2, 0) * 255)
                                new_count_3 += 1

                            # 2
                            elif (class_change == '2'):

                                image_save = images_no3_3[idx // 2]
                                
                                # Save a copy of no_3
                                if (class_idx_store[idx] == '3') and (idx % 2) == 0:
                                    cv2.imwrite('Train_Case/datasets/1/no_3/{}.png'.format(new_count_no3), image_save.numpy().transpose(1, 2, 0) * 255)
                                    new_count_no3 += 1
                                
                                if idx % 2 == 0:
                                    image_save = image_save[:, :112, :]
                                    cv2.imwrite('Train_Case/datasets/2/2/{}.png'.format(new_count_2), image_save.numpy().transpose(1, 2, 0) * 255)
                                    new_count_2 += 1
                                
                                elif idx % 2 == 1:
                                    image_save = image_save[:, 112:, :]
                                    cv2.imwrite('Train_Case/datasets/2/2/{}.png'.format(new_count_2), image_save.numpy().transpose(1, 2, 0) * 255)
                                    new_count_2 += 1
                            
                            # 1
                            elif (class_change == '1'):

                                image_save = images_no3_3[idx // 2]
                                
                                # Save a copy of no_3
                                if (class_idx_store[idx] == '3') and (idx % 2) == 0:
                                    cv2.imwrite('Train_Case/datasets/1/no_3/{}.png'.format(new_count_no3), image_save.numpy().transpose(1, 2, 0) * 255)
                                    new_count_no3 += 1
                                
                                if idx % 2 == 0:
                                    image_save = image_save[:, :112, :]
                                    cv2.imwrite('Train_Case/datasets/2/1/{}.png'.format(new_count_1), image_save.numpy().transpose(1, 2, 0) * 255)
                                    new_count_1 += 1
                                
                                elif idx % 2 == 1:
                                    image_save = image_save[:, 112:, :]
                                    cv2.imwrite('Train_Case/datasets/2/1/{}.png'.format(new_count_1), image_save.numpy().transpose(1, 2, 0) * 255)
                                    new_count_1 += 1
                            
                            # 0
                            elif (class_change == '0'):
                                
                                image_save = images_no3_3[idx // 2]
                                
                                # Save a copy of no_3
                                if (class_idx_store[idx] == '3') and (idx % 2) == 0:
                                    cv2.imwrite('Train_Case/datasets/1/no_3/{}.png'.format(new_count_no3), image_save.numpy().transpose(1, 2, 0) * 255)
                                    new_count_no3 += 1
                                
                                if idx % 2 == 0:
                                    image_save = image_save[:, :112, :]
                                    cv2.imwrite('Train_Case/datasets/2/0/{}.png'.format(new_count_0), image_save.numpy().transpose(1, 2, 0) * 255)
                                    new_count_0 += 1
                                
                                elif idx % 2 == 1:
                                    image_save = image_save[:, 112:, :]
                                    cv2.imwrite('Train_Case/datasets/2/0/{}.png'.format(new_count_0), image_save.numpy().transpose(1, 2, 0) * 255)
                                    new_count_0 += 1
                            
                    return window.destroy()
                
                tk.Button(master=window, text="Submit", command=record).grid(row=13, columnspan=2)

                window.mainloop()


        ########################################################################## Plot Original Points ##########################################################################
        elif key == ord('o'):
            print('\nPlot original points!\n')
            index_store = []
            p = p_original.copy()
            mv = mv_original.copy()
            resize_image = clone.copy()
            for index, pair in enumerate(p):
                resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))
        
        
        ########################################################################## Plot Points ##########################################################################
        elif key == ord('p'):
            print('\nPlot points!\n')
            for index, pair in enumerate(p):
                if index in index_store:
                    continue
                else:
                    resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                    resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))


        ########################################################################## Next/Close ##########################################################################
        elif key == ord('q'):
            print('\nQuiting!\n')
            break


        ########################################################################## Remove Points ##########################################################################
        elif key == ord('r'):
            print('\nRemove points!\n')
            resize_image = clone.copy()


        ########################################################################## Save Keypoints Annotations ##########################################################################
        elif key == ord('s'):
            print('\nSaving!\n')
            
            # Check Annotation Path Exist
            if os.path.exists('Train_Case/COCO/annotations'):
                pass
            else:
                os.makedirs('Train_Case/COCO/annotations')

            # Check Image Path Exist
            if os.path.exists('Train_Case/COCO/default'):
                last_count = len(os.listdir('Train_Case/COCO/default'))
                new_count = last_count + 1
            else:
                os.makedirs('Train_Case/COCO/default')

            # Check Annotation File Exist
            if os.path.isfile('Train_Case/COCO/annotations/person_keypoints_default.json'):
                pass
            else:
                new_dict = {}
                keys = ['licenses', 'info', 'categories', 'images', 'annotations']
                values = [
                    [{'name': '', 'id': 0, 'url': ''}],
                    {'contributor': '', 'date_created': '', 'description': '', 'url': '', 'version': '', 'year': ''},
                    [{'id': 1, 'name': 'Point', 'supercategory': '', 'keypoints': [], 'skeleton': []}],
                    [],
                    []
                    ]
                pair = list(zip(keys, values))

                for key, value in pair:
                    new_dict[key] = value

                with open('Train_Case/COCO/annotations/person_keypoints_default.json', 'w') as f:
                    json.dump(new_dict, f, skipkeys=True)
                

            # Save Image Training Copy
            try:
                cv2.imwrite('Train_Case/COCO/default/{}.png'.format(new_count), image)
            except UnboundLocalError:
                print('Create New Count!')
                new_count = 1
                cv2.imwrite('Train_Case/COCO/default/{}.png'.format(new_count), image)

            # Image Properties
            height = image.shape[0]
            width = image.shape[1]

            image_key = [
                'id',
                'width',
                'height',
                'file_name',
                'license',
                'flickr_url',
                'coco_url',
                'date_captured'
            ]

            image_value = [
                new_count,
                width,
                height,
                '{}.png'.format(new_count),
                0,
                '',
                '',
                0
            ]

            image_list = list(zip(image_key, image_value))
            
            image_dict = {}
            for image_attrib in image_list:
                key = image_attrib[0]
                value = image_attrib[1]
                image_dict[key] = value            

            # Annotation Properties
            ## Keypoints
            keypoints = []
            visibility = np.full(shape=p.shape[0], fill_value=2).astype('float64')
            p_ser = p.copy().astype('float64')
            
            for no_visibility in index_store:
                visibility[no_visibility] = 0

            ann_p = list(zip(p_ser, visibility))
            
            for point, vis in ann_p:
                keypoints.append(point[0])
                keypoints.append(point[1])
                keypoints.append(vis)

            ## Coordinates
            y_cord = p_ser[..., 1]
            x_cord = p_ser[..., 0]

            max_y = np.max(y_cord)
            min_y = np.min(y_cord)
            y_height = max_y - min_y

            max_x = np.max(x_cord)
            min_x = np.min(x_cord)
            x_width = max_x - min_x

            ## BBox
            bbox = [min_x, min_y, x_width, y_height]

            ## Area
            area = x_width * y_height

            ann_key = [
                'id',
                'image_id',
                'category_id',
                'segmentation',
                'area',
                'bbox',
                'iscrowd',
                'attributes',
                'keypoints',
                'num_keypoints'
            ]

            ann_value = [
                new_count,
                new_count,
                1,
                [],
                area,
                bbox,
                0,
                {'occluded': False},
                keypoints,
                12
            ]

            ann_list = list(zip(ann_key, ann_value))

            ann_dict = {}
            for ann_attrib in ann_list:
                key = ann_attrib[0]
                value = ann_attrib[1]
                ann_dict[key] = value
            
            # Save Annotation
            with open('Train_Case/COCO/annotations/person_keypoints_default.json', 'r') as f:
                j = json.load(f)
                j['images'].append(image_dict)
                j['annotations'].append(ann_dict)

            with open('Train_Case/COCO/annotations/person_keypoints_default.json', 'w') as f:
                json.dump(j, f)


        ########################################################################## Visibility ##########################################################################
        elif key == ord('v'):
            print('\nVisibility\n')
            window = tk.Tk()
            window.title('Visibility')
            window.minsize()

            tk.Label(master=window, text='Visibility').pack()

            def change_val():
                global index_store

                var_store = []
                index_store = []
                var_store.append(point_1.get())
                var_store.append(point_2.get())
                var_store.append(point_3.get())
                var_store.append(point_4.get())
                var_store.append(point_5.get())
                var_store.append(point_6.get())
                var_store.append(point_7.get())
                var_store.append(point_8.get())
                var_store.append(point_9.get())
                var_store.append(point_10.get())
                var_store.append(point_11.get())
                var_store.append(point_12.get())
                
                index_store = [index for (index, var) in enumerate(var_store) if var == 1]
                return

            def display():
                global resize_image

                resize_image = clone.copy()
                for index, pair in enumerate(p):
                    if index in index_store:
                        continue
                    else:
                        resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                        resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))
                return cv2.imshow(windowName, resize_image)


            point_1 = tk.IntVar()
            point_2 = tk.IntVar()
            point_3 = tk.IntVar()
            point_4 = tk.IntVar()
            point_5 = tk.IntVar()
            point_6 = tk.IntVar()
            point_7 = tk.IntVar()
            point_8 = tk.IntVar()
            point_9 = tk.IntVar()
            point_10 = tk.IntVar()
            point_11 = tk.IntVar()
            point_12 = tk.IntVar()

            c1 = tk.Checkbutton(window, text='Point 1', variable=point_1, onvalue=1, offvalue=0, command=change_val)
            c1.pack()
            c2 = tk.Checkbutton(window, text='Point 2', variable=point_2, onvalue=1, offvalue=0, command=change_val)
            c2.pack()
            c3 = tk.Checkbutton(window, text='Point 3', variable=point_3, onvalue=1, offvalue=0, command=change_val)
            c3.pack()
            c4 = tk.Checkbutton(window, text='Point 4', variable=point_4, onvalue=1, offvalue=0, command=change_val)
            c4.pack()
            c5 = tk.Checkbutton(window, text='Point 5', variable=point_5, onvalue=1, offvalue=0, command=change_val)
            c5.pack()
            c6 = tk.Checkbutton(window, text='Point 6', variable=point_6, onvalue=1, offvalue=0, command=change_val)
            c6.pack()
            c7 = tk.Checkbutton(window, text='Point 7', variable=point_7, onvalue=1, offvalue=0, command=change_val)
            c7.pack()
            c8 = tk.Checkbutton(window, text='Point 8', variable=point_8, onvalue=1, offvalue=0, command=change_val)
            c8.pack()
            c9 = tk.Checkbutton(window, text='Point 9', variable=point_9, onvalue=1, offvalue=0, command=change_val)
            c9.pack()
            c10 = tk.Checkbutton(window, text='Point 10', variable=point_10, onvalue=1, offvalue=0, command=change_val)
            c10.pack()
            c11 = tk.Checkbutton(window, text='Point 11', variable=point_11, onvalue=1, offvalue=0, command=change_val)
            c11.pack()
            c12 = tk.Checkbutton(window, text='Point 12', variable=point_12, onvalue=1, offvalue=0, command=change_val)
            c12.pack()

            tk.Button(master=window, text='Submit', command=display).pack()

            window.mainloop()

    # Return to main window
    cv2.destroyAllWindows()

    print('\nNot visible points\n', [x+1 for x in index_store])

########################################################################## Mouse Callback Functions ##########################################################################
def CallBackFunc(event, x, y, flags, param):
    global row, p, resize_image

    # Left click to change position
    if event == cv2.EVENT_RBUTTONDOWN:
        # Tkinter
        window = tk.Tk()

        tk.Label(master=window, text='Enter Point Index').pack()
        entry = tk.Entry(master=window)
        entry.pack()

        def change_pos():
            global p, resize_image

            index = int(entry.get())
            p[index-1][:2] = [x, y]
            resize_image = clone.copy()
           
            for index, pair in enumerate(p):
                if index in index_store:
                    continue
                else:
                    resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                    resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))

            cv2.imshow(windowName, resize_image)
            return window.destroy()

        btn = tk.Button(master=window, text='Submit', command=change_pos)
        btn.pack()
        window.mainloop()

    # Drag and Drop
    if event == cv2.EVENT_LBUTTONDOWN:
        row = np.where(np.linalg.norm(p - np.array([x, y]), axis=1) < 10)[0]
        assert row.shape[0] == 1, 'There may be more than 1 or none index in array!' 
        row = row[0]

    if event == cv2.EVENT_LBUTTONUP:
        assert type(row) == np.int64, 'None selected!'
        p[row][:2] = [x, y]
        resize_image = clone.copy()
        row = None

        for index, pair in enumerate(p):
            if index in index_store:
                continue
            else:
                resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))
        cv2.imshow(windowName, resize_image)


def main(ipath, hr_cpath, res_cpath_1, res_cpath_2, view, device):
    global windowName, image, height, width, clone, resize_image, src, dst, p, mv, p_original, mv_original, mSASSS

    mSASSS = {}
    
    for image_name in os.listdir(ipath):
        print('\nCurrent File: {}\n'.format(image_name))

        # Image
        image = cv2.imread(os.path.join(ipath, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[0], image.shape[1]

        # Prediction
        simplehrnet = SimpleHRNet(c=48, key=12, checkpoint_path=hr_cpath, device=torch.device(device))
        p, mv = simplehrnet.predict_single(image)
        p = p[0]
        mv = mv[0]

        # Affine Transformation
        src = np.float32([[0, 0], [width, 0], [0, height]])
        dst = np.float32([[0, 0], [view[0], 0], [0, view[1]]])
        m = cv2.getAffineTransform(src=src, dst=dst)
        resize_image = cv2.warpAffine(image, M=m, dsize=(1000, 1000))

        for pair in p:
            pair[:2] = affine_transform(pair, m, train=False)

        # Cache   
        clone = resize_image.copy()
        p_original, mv_original = p.copy(), mv.copy()

        # CV2 Window
        windowName = "Keypoint Image"
        cv2.namedWindow(windowName)
        cv2.moveWindow(windowName, 500, 200)

        # Tkinter (GUI)
        app(device, res_cpath_1, res_cpath_2)               

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipath', '-ip', help='path to image folder', type=str,  default='datasets/COCO/default')
    parser.add_argument('--res_cpath_1', '-rcp1', help='path to resnet checkpoint for no3, 3', type=str,  default='logs/3_No3/130423_100946/checkpoint_best_0.2344_0.8750.pth')
    parser.add_argument('--res_cpath_2', '-rcp2', help='path to resnet checkpoint for 0, 1, 2', type=str,  default='logs/0_1_2/110423_150447/checkpoint_best_0.2873_0.9748.pth')
    parser.add_argument('--hr_cpath', '-hcp', help='path to hrnet checkpoint', type=str,  default='logs/kp/checkpoint_best_acc_0.9928728138145647.pth')
    parser.add_argument('--view', '-v', help='viewing dimension', type=List[int], default=[1000, 1000])
    parser.add_argument('--device', '-d', help='device', type=str, default='cpu')
    args = parser.parse_args()

    main(**args.__dict__)