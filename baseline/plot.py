import os
import cv2
import pdb
import urlopen
import traceback
import urllib
import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
            w     = float(abs(x_max - x_min))
            h     = float(abs(y_max - y_min)) 
            class_ = int(row_tmp[5])
            try:
                conf = float(row_tmp[6])
            except:
                pass
            if image_file in annotations:
                annotations[image_file]['bboxes'].append([x_min, y_min, w, h])
                annotations[image_file]['category_id'].append(class_)
            else:
                image_path = os.path.join(dir_images, image_file)
                annotations[image_file] = {'image_name' : image_file,'image' : image_path, 'bboxes' : [[x_min, y_min, w, h]], 'category_id' : [class_]}
            
            # if i > 10:
            #     break
    # pdb.set_trace()

    return annotations

BOX_COLOR_TRUE = (0, 255, 0)
BOX_COLOR_PRED = (255, 0, 0)

TEXT_COLOR_TRUE = (0, 0, 0)
TEXT_COLOR_PRED = (255, 255, 255)

def visualize_bbox(img, bbox, class_id, class_idx_to_name, color_box, color_text, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color_box, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color_box, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, lineType=cv2.LINE_AA)
    return img

def visualize(annotations_true, annotations_pred, category_id_to_name, save_dir='./plots'):

    if (1):
        # img = annotations['image'].copy()
        img_true = cv2.imread(annotations_true['image'])
        for idx, bbox in enumerate(annotations_true['bboxes']):
            img_true = visualize_bbox(img_true, bbox, annotations_true['category_id'][idx], category_id_to_name, BOX_COLOR_TRUE, TEXT_COLOR_TRUE)
        
    if (1):
        img_pred = cv2.imread(annotations_pred['image'])
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
    plt.savefig('{0}/{1}'.format(save_dir, annotations_pred['image_name'].split('.')[0] + '.png'), bbox_inches = 'tight',pad_inches = 0, format='png')
    plt.close()

def download_image(url):
    request = urllib.request.Request(url)
    opener = urllib.request.build_opener()
    response = opener.open(request)
    contents = response.read()
    data = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if __name__ == "__main__":

    if (0):
        BOX_COLOR = (255, 0, 0)
        TEXT_COLOR = (255, 255, 255)
        category_id_to_name = {17: 'cat', 18: 'dog'}
        image = download_image('http://images.cocodataset.org/train2017/000000386298.jpg')
        # bboxes = [[xmin, ymin, width, height]]
        annotations = {'image': image, 'bboxes': [[366.7, 80.84, 132.8, 181.84], [5.66, 138.95, 147.09, 164.88]], 'category_id': [18, 17]}
    
    if (0):
        csv_file_true = '../../visum_data/train/annotation.csv'
        csv_file_pred = './predictions.csv'
        dir_images = '../../visum_data/test'
    else:
        csv_file_true = '/home/master/dataset/train/annotation.csv' 
        csv_file_pred = '/home/visum/mody/models/model__frcnn_resnet__iter50.csv'
        dir_images = '/home/master/dataset/test'
    
    if (1):
        annotations_images_true = getannotations(csv_file_true, dir_images)
        annotations_images_pred = getannotations(csv_file_pred, dir_images)
        category_id_to_name = {
                0: 'book', 1: 'bottle', 2: 'box', 3: 'cellphone',
                4: 'cosmetics', 5: 'glasses', 6: 'headphones', 7: 'keys',
                8: 'wallet', 9: 'watch', -1: 'n.a.'}
        
        print (' - [Plot] Finished reading annotations - true :  {0}'.format(csv_file_true))
        print (' - [Plot] Finished reading annotations - pred :  {0}'.format(csv_file_pred))

    if (1):
        
        plot_dir = './plots_50'
        print (' - [Plot] Saving in {0}'.format(plot_dir))
        # with tqdm.tqdm_notebook(total=len(annotations_images)) as pbar:
        with tqdm.tqdm(total=len(annotations_images_pred)) as pbar:
            for i,image in enumerate(annotations_images_pred):
                try:
                    pbar.update(1)
                    annotations_image_true = annotations_images_true[image]
                    annotations_image_pred = annotations_images_pred[image]
                    visualize(annotations_image_true, annotations_image_pred, category_id_to_name, save_dir=plot_dir)
                    # if i > 10:
                    #     break
                except:
                    traceback.print_exc()
                    pass