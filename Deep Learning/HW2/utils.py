import os
import cv2
import glob
import numpy as np
from tqdm import tqdm, tnrange
import random
import gdown
import pandas as pd
import shutil
import matplotlib.pyplot as plt

# sharing_link = "https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view"
# gdown.download(url=sharing_link, output="./CelebAMask-HQ.zip", quiet=False, fuzzy=True)
# !unzip ./CelebAMask-HQ.zip

#if this code doesn't work, you need to download this dataset

def downloading_tts():
    '''
    This function dwonloading the provided celeb_train_test_split.csv files
    make two list for train and test images
    :return: pd.DataFrame, list. // list with lists
    '''
    sharing_link = "https://drive.google.com/file/d/1vO4mJ08FMdcom2-sNs4WHxPztIRYrFYI/view"
    gdown.download(url=sharing_link, output="./celeb_train_test_split.csv", quiet=False, fuzzy=True)
    df = pd.read_csv('./celeb_train_test_split.csv')
    display(df)

    # mask_is_train = df['is_train']==True
    train_files = df[df['is_train']==True]['name'].tolist()
    test_files = df[df['is_train']==False]['name'].tolist()
    return df, [train_files, test_files]

def make_folder(path):
    '''
    This function take a path with desired folder
    and create this if it not exists.
    :param path: str;
    :return: None.
    '''
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))
            
def make_tts(df, files, folders = ['train_img', 'test_img']):
    '''
    This function take
    :param df
    '''
    images = os.listdir('CelebAMask-HQ/CelebA-HQ-img')
    train_files = files[0]
    test_files = files[1]

    make_folder(folders[0])
    make_folder(folders[1])

    for k in tnrange(len(images)):
        if images[k] in train_files:
            source_file = os.path.join('CelebAMask-HQ/CelebA-HQ-img', images[k])
            target_file = os.path.join(folders[0], images[k])
            img = cv2.imread(source_file)
            img = cv2.resize(img, (512,512))
            cv2.imwrite(target_file, img)
        elif images[k] in test_files:
            source_file = os.path.join('CelebAMask-HQ/CelebA-HQ-img', images[k])
            target_file = os.path.join(folders[1], images[k])
            img = cv2.imread(source_file)
            img = cv2.resize(img, (512,512))
            cv2.imwrite(target_file, img)
        

def make_mask(folder_base, folder_save, img_path):
    label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                  'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                  'u_lip', 'l_lip', 'hair', 'hat',
                  'ear_r', 'neck_l', 'neck', 'cloth']    
    mask_img = os.listdir(img_path)
    if '.ipynb_checkpoints' in mask_img:
        mask_img.remove('.ipynb_checkpoints')
    make_folder(folder_save)

    for k in tnrange(len(mask_img)):
        im_base = np.zeros((512, 512))
        folder_num = str(int(mask_img[k].replace('.jpg',''))//2000)
        for idx, label in enumerate(label_list):
            filename = os.path.join(folder_base, folder_num,
                                    mask_img[k].replace('.jpg','').rjust(5,'0')+f'_{label}.png')
            
            if (os.path.exists(filename)):
                im = cv2.imread(filename)
                im = im[:, :, 0]
                im_base[im != 0] = (idx + 1)

        filename_save = os.path.join(folder_save, mask_img[k].replace('.jpg','.png'))
        cv2.imwrite(filename_save, im_base)
        
def conf_mat_vis(conf_mat, y_true, label_list = ['background','skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                  'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                  'u_lip', 'l_lip', 'hair', 'hat',
                  'ear_r', 'neck_l', 'neck', 'cloth']):
    fig, ax = plt.subplots(figsize=(8,8), dpi = 150)
    im = ax.imshow(conf_mat)

    label_list = ['background','skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                      'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                      'u_lip', 'l_lip', 'hair', 'hat',
                      'ear_r', 'neck_l', 'neck', 'cloth'] 

    new_label_list = []
    for i in range(len(label_list)):
        if i in np.unique(y_true):
            new_label_list.append(label_list[i])


    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(new_label_list)), labels=new_label_list)
    ax.set_yticks(np.arange(len(new_label_list)), labels=new_label_list)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            text = ax.text(j, i, conf_mat[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusion_matrix")
    fig.tight_layout()
    plt.show()