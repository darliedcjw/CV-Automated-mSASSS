import os
import pandas as pd
import numpy as np
import tkinter as tk
import cv2
import re
import shutil
import argparse
import torch
from misc.utils import affine_transform, get_angle
from inference import SimpleHRNet

'''
q: quit
s: edit mSASSS scores
r: clear points
o: original points
p: plot points
'''

def app():
    global image, resize_image, clone, p, mv, p_original, mv_original, src, dst, csv_mSASSS, spine

    while(True):
        cv2.setMouseCallback(windowName, CallBackFunc)
        cv2.imshow(windowName, resize_image)
        
        key = cv2.waitKey(0)
        
        if key == ord('r'):

            print('\nRemove points!\n')

            resize_image = clone.copy()

        # Plot Points
        elif key == ord('p'):

            print('\nPlot points!\n')

            for index, pair in enumerate(p):
                resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))

        # Plot Original Points
        elif key == ord('o'):

            print('\nPlot original points!\n')

            p = p_original.copy()
            mv = mv_original.copy()
            resize_image = clone.copy()
            for index, pair in enumerate(p):
                resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))

        elif key == ord('b'):

            print('\nDisplay bounding boxes!\n')

            avg_len = abs(p[3] - p[5]).mean()
            for index, pair in enumerate(p):
                if index % 2 == 1:
                    pt1 = (int(pair[0] - avg_len), int(pair[1] - avg_len))
                    pt2 = (int(pair[0] + avg_len), int(pair[1] + avg_len))
                    resize_image = cv2.rectangle(resize_image, pt1=pt1, pt2=pt2, color=(255, 255, 255), thickness=1)
            else:
                continue

        # Save
        elif key == ord('s'):
            
            print('\nSaving!\n')

            if len(mSASSS) == 0:

                window = tk.Tk()
                window.title('CSV mSASSS Score')
                window.minsize(width=200, height=300)
                window.grid_columnconfigure(tuple(range(12)), weight=1)

                if spine == "CS":
                    print('\nUsing CS scores!\n')
                    score_1 = csv_mSASSS['C2 lower border']
                    score_2 = csv_mSASSS['C3 upper border']
                    score_3 = csv_mSASSS['C3 lower border']
                    score_4 = csv_mSASSS['C4 upper border']
                    score_5 = csv_mSASSS['C4 lower border']
                    score_6 = csv_mSASSS['C5 upper border']
                    score_7 = csv_mSASSS['C5 lower border']
                    score_8 = csv_mSASSS['C6 upper border']
                    score_9 = csv_mSASSS['C6 lower border']
                    score_10 = csv_mSASSS['C7 upper border']
                    score_11 = csv_mSASSS['C7 lower border']
                    score_12 = csv_mSASSS['T1 upper border'] 

                elif spine == "LS":
                    print('\nUsing LS scores!\n')
                    score_1 = csv_mSASSS['T12 lower border']
                    score_2 = csv_mSASSS['L1 upper border']
                    score_3 = csv_mSASSS['L1 lower border']
                    score_4 = csv_mSASSS['L2 upper border']
                    score_5 = csv_mSASSS['L2 lower border']
                    score_6 = csv_mSASSS['L3 upper border']
                    score_7 = csv_mSASSS['L3 lower border']
                    score_8 = csv_mSASSS['L4 upper border']
                    score_9 = csv_mSASSS['L4 lower border']
                    score_10 = csv_mSASSS['L5 upper border']
                    score_11 = csv_mSASSS['L5 lower border']
                    score_12 = csv_mSASSS['S1 upper border'] 
                

                tk.Label(master=window, text="mSASSS Score").grid(row=0, columnspan=2, sticky='ew')
                tk.Label(master=window, text="Point 1").grid(row=1, column=0)
                entry1 = tk.Entry(master=window)
                entry1.insert(0, score_1)
                entry1.grid(row=1, column=1)
                tk.Label(master=window, text="Point 2").grid(row=2, column=0)
                entry2 = tk.Entry(master=window)
                entry2.insert(0, score_2)
                entry2.grid(row=2, column=1) 
                tk.Label(master=window, text="Point 3").grid(row=3, column=0)
                entry3 = tk.Entry(master=window)
                entry3.insert(0, score_3)
                entry3.grid(row=3, column=1)           
                tk.Label(master=window, text="Point 4").grid(row=4, column=0)
                entry4 = tk.Entry(master=window)
                entry4.insert(0, score_4)
                entry4.grid(row=4, column=1)
                tk.Label(master=window, text="Point 5").grid(row=5, column=0)
                entry5 = tk.Entry(master=window)
                entry5.insert(0, score_5)
                entry5.grid(row=5, column=1)      
                tk.Label(master=window, text="Point 6").grid(row=6, column=0)
                entry6 = tk.Entry(master=window)
                entry6.insert(0, score_6)
                entry6.grid(row=6, column=1)
                tk.Label(master=window, text="Point 7").grid(row=7, column=0)
                entry7 = tk.Entry(master=window)
                entry7.insert(0, score_7)
                entry7.grid(row=7, column=1)
                tk.Label(master=window, text="Point 8").grid(row=8, column=0)
                entry8 = tk.Entry(master=window)
                entry8.insert(0, score_8)
                entry8.grid(row=8, column=1)
                tk.Label(master=window, text="Point 9").grid(row=9, column=0)
                entry9 = tk.Entry(master=window)
                entry9.insert(0, score_9)
                entry9.grid(row=9, column=1)
                tk.Label(master=window, text="Point 10").grid(row=10, column=0)
                entry10 = tk.Entry(master=window)
                entry10.insert(0, score_10)
                entry10.grid(row=10, column=1)
                tk.Label(master=window, text="Point 11").grid(row=11, column=0)
                entry11 = tk.Entry(master=window)
                entry11.insert(0, score_11)
                entry11.grid(row=11, column=1)
                tk.Label(master=window, text="Point 12").grid(row=12, column=0)
                entry12 = tk.Entry(master=window)
                entry12.insert(0, score_12)
                entry12.grid(row=12, column=1)

                mSASSS['score_1'] = entry1.get()
                mSASSS['score_2'] = entry2.get()
                mSASSS['score_3'] = entry3.get()
                mSASSS['score_4'] = entry4.get()
                mSASSS['score_5'] = entry5.get()
                mSASSS['score_6'] = entry6.get()
                mSASSS['score_7'] = entry7.get()
                mSASSS['score_8'] = entry8.get()
                mSASSS['score_9'] = entry9.get()
                mSASSS['score_10'] = entry10.get()
                mSASSS['score_11'] = entry11.get()
                mSASSS['score_12'] = entry12.get()

            else:
                window = tk.Tk()
                window.title('mSASSS Score')
                window.minsize(width=200, height=300)
                window.grid_columnconfigure((0, 1), weight=1)

                score_1 = mSASSS['score_1']
                score_2 = mSASSS['score_2']
                score_3 = mSASSS['score_3']
                score_4 = mSASSS['score_4']
                score_5 = mSASSS['score_5']
                score_6 = mSASSS['score_6']
                score_7 = mSASSS['score_7']
                score_8 = mSASSS['score_8']
                score_9 = mSASSS['score_9']
                score_10 = mSASSS['score_10']
                score_11 = mSASSS['score_11']
                score_12 = mSASSS['score_12']

                tk.Label(master=window, text="mSASSS Score").grid(row=0, columnspan=2, sticky='ew')
                tk.Label(master=window, text="Point 1").grid(row=1, column=0)
                entry1 = tk.Entry(master=window)
                entry1.insert(0, score_1)
                entry1.grid(row=1, column=1)
                tk.Label(master=window, text="Point 2").grid(row=2, column=0)
                entry2 = tk.Entry(master=window)
                entry2.insert(0, score_2)
                entry2.grid(row=2, column=1) 
                tk.Label(master=window, text="Point 3").grid(row=3, column=0)
                entry3 = tk.Entry(master=window)
                entry3.insert(0, score_3)
                entry3.grid(row=3, column=1)           
                tk.Label(master=window, text="Point 4").grid(row=4, column=0)
                entry4 = tk.Entry(master=window)
                entry4.insert(0, score_4)
                entry4.grid(row=4, column=1)
                tk.Label(master=window, text="Point 5").grid(row=5, column=0)
                entry5 = tk.Entry(master=window)
                entry5.insert(0, score_5)
                entry5.grid(row=5, column=1)      
                tk.Label(master=window, text="Point 6").grid(row=6, column=0)
                entry6 = tk.Entry(master=window)
                entry6.insert(0, score_6)
                entry6.grid(row=6, column=1)
                tk.Label(master=window, text="Point 7").grid(row=7, column=0)
                entry7 = tk.Entry(master=window)
                entry7.insert(0, score_7)
                entry7.grid(row=7, column=1)
                tk.Label(master=window, text="Point 8").grid(row=8, column=0)
                entry8 = tk.Entry(master=window)
                entry8.insert(0, score_8)
                entry8.grid(row=8, column=1)
                tk.Label(master=window, text="Point 9").grid(row=9, column=0)
                entry9 = tk.Entry(master=window)
                entry9.insert(0, score_9)
                entry9.grid(row=9, column=1)
                tk.Label(master=window, text="Point 10").grid(row=10, column=0)
                entry10 = tk.Entry(master=window)
                entry10.insert(0, score_10)
                entry10.grid(row=10, column=1)
                tk.Label(master=window, text="Point 11").grid(row=11, column=0)
                entry11 = tk.Entry(master=window)
                entry11.insert(0, score_11)
                entry11.grid(row=11, column=1)
                tk.Label(master=window, text="Point 12").grid(row=12, column=0)
                entry12 = tk.Entry(master=window)
                entry12.insert(0, score_12)
                entry12.grid(row=12, column=1)

            # Function
            def record():
                global mSASSS
                mSASSS['score_1'] = entry1.get()
                mSASSS['score_2'] = entry2.get()
                mSASSS['score_3'] = entry3.get()
                mSASSS['score_4'] = entry4.get()
                mSASSS['score_5'] = entry5.get()
                mSASSS['score_6'] = entry6.get()
                mSASSS['score_7'] = entry7.get()
                mSASSS['score_8'] = entry8.get()
                mSASSS['score_9'] = entry9.get()
                mSASSS['score_10'] = entry10.get()
                mSASSS['score_11'] = entry11.get()
                mSASSS['score_12'] = entry12.get()
                return window.destroy()

            tk.Button(master=window, text="Submit", command=record).grid(row=13, columnspan=2)
            window.mainloop() 


        # Close
        elif key == ord('q'):
            print('\nQuiting!\n')
            break
    
    cv2.destroyAllWindows()

    print('\nFinal mSASSS scores!\n', mSASSS)

# Callback function
def CallBackFunc(event, x, y, flags, param):
    global windowName, p, row, resize_image

    # Left click to change position
    if event == cv2.EVENT_RBUTTONDOWN:
        # Tkinter
        window = tk.Tk()

        tk.Label(master=window, text='Enter Point Index').pack()
        entry = tk.Entry(master=window)
        entry.pack()

        def change_pos():
            global clone, p, resize_image

            index = int(entry.get())
            p[index-1][:2] = [x, y]
            resize_image = clone.copy()
           
            for index, pair in enumerate(p):
                resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))

            cv2.imshow(windowName, resize_image)
            return window.destroy()

        btn = tk.Button(master=window, text='Submit', command=change_pos)
        btn.pack()
        window.mainloop()

    if event == cv2.EVENT_LBUTTONDOWN:
        row = np.where(np.linalg.norm(p - np.array([x, y]), axis=1) < 10)[0]
        assert row.shape[0] == 1, 'there may be more than 1 or none index in array!' 
        row = row[0]

    if event == cv2.EVENT_LBUTTONUP:
        assert type(row) == np.int64, 'none selected!'
        p[row][:2] = [x, y]
        resize_image = clone.copy()
        row = None

        for index, pair in enumerate(p):
            resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
            resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))
        cv2.imshow(windowName, resize_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipath', '-ip', help='path to image folder', type=str,  default='datasets/COCO/default')
    parser.add_argument('--ppath', '-pp', help='path to patient index excel', type=str, default='datasets/Scores/patient_index.xlsx')
    parser.add_argument('--spath', '-sp', help='path to score csv', type=str, default='datasets/Scores/scores.csv')
    parser.add_argument('--cpath', '-cp', help='path to class folder', type=str, default='image_class')
    parser.add_argument('--device', '-d', help='device', type=str, default='cpu')
    args = parser.parse_args()
    
    image_path = args.__dict__['ipath']
    patient_path = args.__dict__['ppath']
    scores_path = args.__dict__['spath']
    class_path = args.__dict__['cpath']
    device = args.__dict__['device']

    scores = pd.read_csv(scores_path).drop(columns=['Unnamed: 0'])
    patient_index = pd.read_excel(patient_path)

    print('\nScores loaded!\n')
    print('\nPatient index loaded!\n')

    for image_name in os.listdir(image_path):

        print('\nCurrent File: {}\n'.format(image_name))
        
        mSASSS = {}
        csv_mSASSS = {}

        '''
        Score
        '''
        patient_score = scores.copy()

        old = r'(\d{2})(\d{2})(\d{4})'
        new = r'\3-\2-\1'

        date = re.sub(
            old,
            new,
            image_name.split('_')[1]
            )

        ashi_index = int(
            image_name.split('_')[0]
            )

        spine = str.upper(image_name.split('_')[2])
        nric = patient_index[patient_index['Subject ID'].apply(lambda x: int(x.split('_')[-1])) == ashi_index]['NRIC'].values[0]
        print('\n', nric, '\n')

        patient = zip(
            ['Patient ID', 'Date of X-Ray'],
            [nric, date]
            )

        for column_id, value in patient:
            patient_score = patient_score[patient_score[column_id] == value]
        
        print('\nRetrieved DataFrame!\n', patient_score)

        na_columns = ('Patient ID', 'mSASSS Score', 'Date of X-Ray', 'Year')
        columns = patient_score.columns

        for column in columns:
            if column in na_columns:
                continue
            else:
                try:
                    csv_mSASSS[column] = int(patient_score[column].values[0])
                except (KeyError, IndexError, ValueError):
                    csv_mSASSS[column] = np.NAN
        
        print('\nmSASSS Scores!\n', csv_mSASSS)

        '''
        Image
        '''
        image = cv2.imread(os.path.join(image_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[0] - 1, image.shape[1] - 1

        # Prediction
        simplehrnet = SimpleHRNet(c=48, key=12, checkpoint_path='./logs/20221220_1651/checkpoint_best_acc_0.9928728138145647.pth', device=torch.device(device))
        p, mv = simplehrnet.predict_single(image)
        p = p[0]
        mv = mv[0]

        # Affine Transformation
        src = np.float32([[0, 0], [width, 0], [0, height]])
        dst = np.float32([[0, 0], [1000, 0], [0, 1000]])
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

        # Tkinter
        app()

        # Reverse Affine
        m_reverse = cv2.getAffineTransform(src=dst, dst=src)
        
        for pair in p:
            pair[:2] = affine_transform(pair, m_reverse, train=False)

        avg_len = abs(p[3] - p[5]).mean()
        keys = list(mSASSS.keys())
        count_nan = 0

        for key in keys:
            count_nan += (mSASSS[key] == 'NaN')

        for index, pair in enumerate(p):
            if index % 2 == 1:
                # Rotation,  Multiply by negative sign since rotation is opposite direction in opencv
                pt1, pt2 = p[index - 1].copy(), pair[:2].copy()
                rot_deg = get_angle(pt1=pt1, pt2=pt2)
        
                # Affine transform for each of the outer points
                m = cv2.getRotationMatrix2D(center=(width // 2, height // 2), angle=rot_deg, scale=1)
                image_rot = cv2.warpAffine(src=image, M=m, dsize=(width, height))
                x, y = affine_transform(pt2, m, train=False)
                rot_width, rot_height = image_rot.shape[1], image_rot.shape[0]

                # Crop and Save
                low_pt = (np.clip(int(x - avg_len), a_min=0, a_max=rot_width), np.clip(int(y - avg_len), a_min=0, a_max=rot_height))
                high_pt = (np.clip(int(x + avg_len), a_min=0, a_max=rot_width), np.clip(int(y + avg_len), a_min=0, a_max=rot_height))

                
                if (mSASSS[keys[index]] == 'NaN') or (mSASSS[keys[index - 1]] == 'NaN'):
                    if count_nan > 6:
                        '''
                        NaN is more than 6, move entire image to unknown folder
                        '''
                        old_path = os.path.join(image_path, image_name)
                        new_path = os.path.join(class_path, 'unknown', image_name)
                        shutil.move(old_path, new_path)
                        break
                    else:
                        '''
                        NaN < 6, move cropped NaN to unknown folder
                        '''
                        image_crop = image_rot[low_pt[1]:high_pt[1] + 1, low_pt[0]:high_pt[0] + 1, :]
                        file_path = '{}/unknown/{}_{}_{}_{}.png'.format(class_path, ashi_index, index-1, date, spine)
                        cv2.namedWindow('crop')
                        cv2.moveWindow('crop', 1000, 200)
                        cv2.imshow('crop', image_crop)
                        cv2.waitKey(0)
                        cv2.imwrite(file_path, image_crop)
                        cv2.destroyAllWindows()


                elif (mSASSS[keys[index]] == '3') or (mSASSS[keys[index - 1]] == '3'):
                    image_crop = image_rot[low_pt[1]:high_pt[1] + 1, low_pt[0]:high_pt[0] + 1, :]
                    file_path = '{}/{}/{}_{}_{}_{}.png'.format(class_path, mSASSS[keys[index-1]], ashi_index, index-1, date, spine)
                    cv2.namedWindow('crop')
                    cv2.moveWindow('crop', 1000, 200)
                    cv2.imshow('crop', image_crop)
                    cv2.waitKey(0)
                    cv2.imwrite(file_path, image_crop)
                    cv2.destroyAllWindows()
                
                else:
                    image_crop = image_rot[low_pt[1]:high_pt[1] + 1, low_pt[0]:high_pt[0] + 1, :]
                    file_path = '{}/no_3/{}_{}_{}_{}.png'.format(class_path ,ashi_index, index-1, date, spine)
                    cv2.namedWindow('crop')
                    cv2.moveWindow('crop', 1000, 200)
                    cv2.imshow('crop', image_crop)
                    cv2.waitKey(0)
                    cv2.imwrite(file_path, image_crop)
                    cv2.destroyAllWindows()

                    image_crop_l = image_rot[int(y - y_buffer):high_pt[1] + 1, low_pt[0]:high_pt[0] + 1, :]
                    image_crop_u = image_rot[low_pt[1] + 1:int(y + y_buffer), low_pt[0]:high_pt[0] + 1, :]
                    
                    file_path_l = '{}/{}/{}_{}_{}_{}.png'.format(class_path, mSASSS[keys[index]], ashi_index, index, date, spine)
                    file_path_u = '{}/{}/{}_{}_{}_{}.png'.format(class_path, mSASSS[keys[index-1]], ashi_index, index-1, date, spine)
                    
                    cv2.namedWindow('crop_lower')
                    cv2.moveWindow('crop_lower', 1000, 200)
                    cv2.imshow('crop_lower', image_crop_l)
                    cv2.waitKey(0)
                    cv2.imwrite(file_path_l, image_crop_l)
                    cv2.destroyAllWindows()

                    cv2.namedWindow('crop_upper')
                    cv2.moveWindow('crop_upper', 1000, 200)
                    cv2.imshow('crop_upper', image_crop_u)
                    cv2.waitKey(0)
                    cv2.imwrite(file_path_u, image_crop_u)
                    cv2.destroyAllWindows()
        
        print('\nShifting!\n')
        if count_nan > 6:
            continue
        else:
            shutil.move(os.path.join(image_path, image_name), 'datasets/COCO/default_done')
            
        print('\n\nNext\n\n')
