import os
import shutil
import random
import cv2
import numpy as np
import json

from tqdm import tqdm


total_class_names_set = set()


def polygon2bbox(polygon, h, w):
    xmin, xmax, ymin, ymax = w, 0, h, 0
    for [ptx, pty] in polygon:
        xmin, xmax = min(xmin, ptx), max(xmax, ptx)
        ymin, ymax = min(ymin, pty), max(ymax, pty)
    return [xmin, xmax, ymin, ymax]
    
def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def convert_label(anno_abs_path, lb_path, image_id, class_names):
    if not os.path.exists(anno_abs_path):
        return image_id + "_None"
        
    in_file = json.load(open(anno_abs_path))
        
    out_file = open(lb_path, 'w')
    
    h, w = in_file["imgHeight"], in_file["imgWidth"]  # 1024, 2048
    label_dict_list = in_file["objects"]
    is_nolabel_flag = True
    for i, label_dict in enumerate(label_dict_list):
        label = label_dict["label"]
        total_class_names_set.add(label)
        if label in class_names:
            is_nolabel_flag = False
            polygon = label_dict["polygon"]
            bbox = polygon2bbox(polygon, h, w)  # 'xmin', 'xmax', 'ymin', 'ymax'
            bb = convert_box((w, h), bbox)
            cls_id = class_names.index(label)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')
    
    if is_nolabel_flag:
        return image_id + "_Nolabel"
    else:
        return None
    

if __name__ == "__main__":
    
    '''
    https://www.cityscapes-dataset.com/downloads/
    https://github.com/mcordts/cityscapesScripts
    '''
    
    class_names = [ 'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck' ]  # number of selected classes, nc=8
    data_root_path = "/datasdc/zhouhuayi/dataset/domain_adaptation/CityScapesFoggy"  # train 2975, val 500


    if os.path.exists(os.path.join(data_root_path, "yolov5_format")):
        shutil.rmtree(os.path.join(data_root_path, "yolov5_format"))
    os.mkdir(os.path.join(data_root_path, "yolov5_format"))
    for image_set in ["train", "val"]:

        imgs_path = os.path.join(data_root_path, "yolov5_format", "images", image_set)
        if not os.path.exists(os.path.join(data_root_path, "yolov5_format", "images")):
            os.mkdir(os.path.join(data_root_path, "yolov5_format", "images"))
        if not os.path.exists(imgs_path):
            os.mkdir(imgs_path)
            
        lbs_path = os.path.join(data_root_path, "yolov5_format", "labels", image_set)
        if not os.path.exists(os.path.join(data_root_path, "yolov5_format", "labels")):
            os.mkdir(os.path.join(data_root_path, "yolov5_format", "labels"))
        if not os.path.exists(lbs_path):
            os.mkdir(lbs_path)
        
        ori_imgs_path = os.path.join(data_root_path, "leftImg8bit_foggyDBF", image_set)
        ori_anno_path = os.path.join(data_root_path.replace("Foggy", ""), "gtFine", image_set)  # annotations are placed in CityScapes
        city_names = os.listdir(ori_imgs_path)
        error_list = []
        for city_name in tqdm(city_names):
            ori_imgs_path_city = os.path.join(ori_imgs_path, city_name)
            ori_anno_path_city = os.path.join(ori_anno_path, city_name)
            for img_name in os.listdir(ori_imgs_path_city):
                # if "0.01" not in img_name: continue
                if "0.02" not in img_name: continue  # we choose the most difficult beta param
                # if "0.005" not in img_name: continue
                img_abs_path = os.path.join(ori_imgs_path_city, img_name)  # old img path
                # id = img_name.replace("_leftImg8bit_foggy_beta_0.01.png", "")
                id = img_name.replace("_leftImg8bit_foggy_beta_0.02.png", "")  # we choose the most difficult beta param
                # id = img_name.replace("_leftImg8bit_foggy_beta_0.005.png", "")
                anno_abs_path = os.path.join(ori_anno_path_city, id+"_gtFine_polygons.json")
                lb_path = os.path.join(lbs_path, id+".txt")  # new label path
                res = convert_label(anno_abs_path, lb_path, id, class_names)  # convert labels to YOLO format
                if os.path.exists(lb_path):
                    # shutil.copy(img_abs_path, os.path.join(imgs_path, id+".jpg"))  # move image
                    os.system("ln -s %s %s"%(img_abs_path, os.path.join(imgs_path, id+".jpg")))  # soft link of image
                if res is not None:
                    error_list.append(res)
            # finished one city
        print(image_set, "--> error_list:", len(error_list), "\n", error_list)
        
    print("[OK] finished one dataset %s"%(data_root_path))
    print("selected class names in CityScapes are: \n %s"%(class_names))
    print("all class names in CityScapes are: \n %s"%(total_class_names_set))

'''
train --> error_list: 10
 ['monchengladbach_000000_015561_Nolabel', 'weimar_000097_000019_Nolabel', 'weimar_000067_000019_Nolabel', 'dusseldorf_000101_000019_Nolabel', 'dusseldorf_000106_000019_Nolabel', 'bochum_000000_031152_Nolabel', 'strasbourg_000000_012934_Nolabel', 'strasbourg_000000_035571_Nolabel', 'strasbourg_000000_036016_Nolabel', 'strasbourg_000000_023854_Nolabel']
val --> error_list: 8
 ['lindau_000040_000019_Nolabel', 'lindau_000045_000019_Nolabel', 'lindau_000019_000019_Nolabel', 'lindau_000021_000019_Nolabel', 'lindau_000049_000019_Nolabel', 'lindau_000017_000019_Nolabel', 'lindau_000018_000019_Nolabel', 'lindau_000032_000019_Nolabel']
[OK] finished one dataset /datasdc/zhouhuayi/dataset/domain_adaptation/CityScapesFoggy
selected class names in CityScapes are:
 ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck']
all class names in CityScapes are:
 {'polegroup', 'license plate', 'car', 'caravan', 'bicyclegroup', 'road', 'motorcyclegroup', 'traffic sign', 'static', 'ego vehicle', 'tunnel', 'train', 'motorcycle', 'building', 'wall', 'pole', 'rider', 'vegetation', 'trailer', 'bicycle', 'rail track', 'terrain', 'parking', 'sidewalk', 'guard rail', 'dynamic', 'bus', 'sky', 'bridge', 'truckgroup', 'cargroup', 'person', 'traffic light', 'ridergroup', 'rectification border', 'fence', 'persongroup', 'out of roi', 'truck', 'ground'} (nc=40)

'''

    
