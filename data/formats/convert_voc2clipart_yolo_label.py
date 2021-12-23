import os
import shutil
import random
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm


def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def convert_label(path, lb_path, image_id, img_path, class_names):
    if not os.path.exists(os.path.join(path, "Annotations", image_id+".xml")):
        return image_id + "_None"
    in_file = open(os.path.join(path, "Annotations", image_id+".xml"))

    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    if w == 0 or h == 0:
        img = cv2.imread(img_path)
        imgh, imgw, imgc = img.shape
        size.find('width').text = str(imgw)
        size.find('height').text = str(imgh)
        tree.write(os.path.join(path, "Annotations", image_id+".xml"))
        return image_id + "_Error"
        
    assert w != 0 and h != 0, str(lb_path) + " has a zero width or zero height --> " + str(w) + "#" + str(h)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        # if cls in class_names and not int(obj.find('difficult').text) == 1:
        if cls in class_names:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = class_names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

    return None

if __name__ == "__main__":
    
    
    
    class_names = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' ]  # number of classes, nc=20
    data_root_path_list = [
        ["/datasdc/zhouhuayi/dataset/domain_adaptation/clipart", "/datasdc/zhouhuayi/dataset/domain_adaptation/clipart"],
        ["/datasdc/zhouhuayi/dataset/domain_adaptation/clipart2voc", "/datasdc/zhouhuayi/dataset/domain_adaptation/clipart2voc"],
        ["/datasdc/zhouhuayi/dataset/domain_adaptation/pascalvoc/VOC2007", "/datasdc/zhouhuayi/dataset/PascalVOC0712/VOCdevkit/VOC2007"],
        ["/datasdc/zhouhuayi/dataset/domain_adaptation/pascalvoc/VOC2012", "/datasdc/zhouhuayi/dataset/PascalVOC0712/VOCdevkit/VOC2012"],
        ["/datasdc/zhouhuayi/dataset/domain_adaptation/dt_clipart/VOC2007", "/datasdc/zhouhuayi/dataset/domain_adaptation/dt_clipart/VOC2007"],
        ["/datasdc/zhouhuayi/dataset/domain_adaptation/dt_clipart/VOC2012", "/datasdc/zhouhuayi/dataset/domain_adaptation/dt_clipart/VOC2012"],
        ]
    
    
    for [data_root_path, data_root_path_ori] in data_root_path_list:

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
            
            txt_file_path = os.path.join(data_root_path_ori, "ImageSets/Main", image_set+".txt")
            image_ids = open(txt_file_path).read().strip().split()
            
            error_list = []
            for id in tqdm(image_ids):
                f = os.path.join(data_root_path_ori, "JPEGImages", id+".jpg")  # old img path
                lb_path = os.path.join(lbs_path, id+".txt")  # new label path
                res = convert_label(data_root_path_ori, lb_path, id, f, class_names)  # convert labels to YOLO format
                if os.path.exists(lb_path):
                    shutil.copy(f, os.path.join(imgs_path, id+".jpg"))  # move image
                    # os.system("ln -s %s %s"%(f, os.path.join(imgs_path, id+".jpg")))  # soft link of image
                if res is not None:
                    error_list.append(res)
            print("error_list:", len(error_list), "\n", error_list)
            
        print("[OK] finished one dataset %s, %s"%(data_root_path, data_root_path_ori))
        
    print("[OK] all finished!")

    
