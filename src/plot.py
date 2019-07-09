import os
import cv2
import pdb
import urlopen
import traceback
import urllib
import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BOX_COLOR_TRUE = (0, 255, 0)
BOX_COLOR_PRED = (255, 0, 0)

TEXT_COLOR_TRUE = (0, 0, 0)
TEXT_COLOR_PRED = (255, 255, 255)

category_id_to_name = {
    0: 'book', 1: 'bottle', 2: 'box', 3: 'cellphone',
    4: 'cosmetics', 5: 'glasses', 6: 'headphones', 7: 'keys',
    8: 'wallet', 9: 'watch', -1: 'n.a.'}

# csv_file = [img_name, x_min, x_max, y_min, y_max, class, conf]
def getannotations(csv_file, dir_images):
    annotations = {}
    with open(csv_file, 'r') as fp:
        for i, row in enumerate(fp.readlines()):
            row_tmp = row.split(',')
            image_file = str(row_tmp[0])
            x_min = float(row_tmp[1])
            y_min = float(row_tmp[2])
            x_max = float(row_tmp[3])
            y_max = float(row_tmp[4]) 
            class_ = int(row_tmp[5])
            try:
                conf = float(row_tmp[6])
            except:
                pass
            if image_file in annotations:
                annotations[image_file]['bboxes'].append([x_min, y_min, x_max, y_max])
                annotations[image_file]['category_id'].append(class_)
            else:
                image_path = os.path.join(dir_images, image_file)
                annotations[image_file] = {'image_name' : image_file,'image' : image_path, 'image_path' : image_path,'bboxes' : [[x_min, y_min, x_max, y_max]], 'category_id' : [class_]}

    return annotations

def visualize_bbox(img, bbox, class_id, class_idx_to_name, color_box, color_text, thickness=2):
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color_box, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color_box, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, lineType=cv2.LINE_AA)
    return img

def visualize(annotations_true, annotations_pred, category_id_to_name, save_dir='./plots'):

    if (1):
        # img = annotations['image'].copy()
        img_true = np.array(Image.open(annotations_true['image']))
        for idx, bbox in enumerate(annotations_true['bboxes']):
            img_true = visualize_bbox(img_true, bbox, annotations_true['category_id'][idx], category_id_to_name, BOX_COLOR_TRUE, TEXT_COLOR_TRUE)
        
    if (1):
        img_pred = np.array(Image.open(annotations_pred['image']))
        for idx, bbox in enumerate(annotations_pred['bboxes']):
            img_pred = visualize_bbox(img_pred, bbox, annotations_pred['category_id'][idx], category_id_to_name, BOX_COLOR_PRED, TEXT_COLOR_PRED)

    f,axarr = plt.subplots(1,2)
    axarr[0].imshow(img_true)
    axarr[0].set_title('Ground Truth')
    axarr[0].set_xticks([])
    axarr[0].axis('off')
    axarr[1].imshow(img_pred)
    axarr[1].set_title('Predictions')
    axarr[1].set_xticks([])
    axarr[1].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if len(save_dir):
        plt.savefig('{0}/{1}'.format(save_dir, annotations_pred['image_name'].split('.')[0] + '.png'), bbox_inches = 'tight',pad_inches = 0, format='png')
    plt.close()

if __name__ == "__main__":
    
    if (1):
        csv_file_true = '../../../visum_data/test/annotation.csv'
        csv_file_pred = '../../../code/competition/predictions/model__frcnn_resnet__iter50.csv'
        dir_images    = '../../../visum_data/test'
        plot_dir      = './plots_50'
        torch_seed    = 1
    else:
        csv_file_true = '/home/master/dataset/train/annotation.csv' 
        csv_file_pred = '/home/visum/mody/models/model__frcnn_resnet__iter50.csv'
        dir_images    = '/home/master/dataset/test'
        plot_dir      = './plots_50'

    if (1):
        annotations_images_true = getannotations(csv_file_true, dir_images)
        annotations_images_pred = getannotations(csv_file_pred, dir_images)
        print (' - [Plot] Finished reading annotations - true :  {0}'.format(csv_file_true))
        print (' - [Plot] Finished reading annotations - pred :  {0}'.format(csv_file_pred))
        print (' - [Plot] Saving in {0}'.format(plot_dir))
        if len(plot_dir):
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)
                    

    if (1):
        with tqdm.tqdm(total=len(annotations_images_pred)) as pbar:
            for i,image in enumerate(annotations_images_pred):
                try:
                    pbar.update(1)
                    annotations_image_true = annotations_images_true[image]
                    annotations_image_pred = annotations_images_pred[image]
                    visualize(annotations_image_true, annotations_image_pred, category_id_to_name, save_dir=plot_dir)
                    if i > 10:
                        break
                except:
                    traceback.print_exc()
                    pass