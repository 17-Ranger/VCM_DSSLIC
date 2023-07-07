<<<<<<< HEAD
import json
import csv
import os
import shutil

# fw=open("COCO_eval/instance_val2017.json","r",encoding='utf-8')

def id_map(label):
    if label == 1:
        return 'person'
    if label == 3:
        return 'car'
    if label == 4:
        return 'motorcycle'
    if label == 5:
        return 'airplane'
    if label == 6:
        return 'bus'
    if label == 7:
        return 'train'
    if label == 8:
        return 'truck'
    if label == 9:
        return 'boat'
    if label == 10:
        return 'traffic_light'
    if label == 16:
        return 'bird'
    if label == 17:
        return 'cat'
    if label == 18:
        return 'dog'
    if label == 19:
        return 'horse'
    if label == 20:
        return 'sheep'
    if label == 22:
        return 'elephant'
    if label == 24:
        return 'zebra'
    if label == 25:
        return 'giraffe'
    if label == 27:
        return 'backpack'
    if label == 31:
        return 'handbag'
    if label == 32:
        return 'tie'
    if label == 33:
        return 'suitcase'
    if label == 41:
        return 'skateboard'
    if label == 42:
        return 'surfboard'
    if label == 43:
        return 'tennis_racket'
    if label == 44:
        return 'bottle'
    if label == 49:
        return 'knife'
    if label == 50:
        return 'spoon'
    if label == 51:
        return 'bowl'
    if label == 53:
        return 'apple'
    if label == 54:
        return 'sandwich'
    if label == 55:
        return 'orange'
    if label == 56:
        return 'broccoli'
    if label == 57:
        return 'carrot'
    if label == 59:
        return 'pizza'
    if label == 61:
        return 'cake'
    if label == 63:
        return 'couch'
    if label == 70:
        return 'toilet'
    if label == 73:
        return 'laptop'
    if label == 84:
        return 'book'
    if label == 85:
        return 'clock'
    if label == 86:
        return 'vase'
    if label == 60:
        return 'donut'
    if label == 47:
        return 'cup'
    if label == 77:
        return 'cell_phone'
    if label == 76:
        return 'keyboard'
    if label == 21:
        return 'cow'
    if label == 32:
        return 'sports_ball'
    else:
        return 1

def trans_anno(anno, images):
    # del anno['segmentation']
    # del anno['area']
    # del anno['id']
    for i in images:
        if i['id'] == anno['image_id']:
            anno['ImageID'] = i['file_name'][:-4]
            anno['h'] = i['height']
            anno['w'] = i['width']

    bbox1 = round(anno['bbox'][0] / anno['w'], 8)
    bbox2 = round(anno['bbox'][1] / anno['h'], 8)
    bbox3 = round((anno['bbox'][0] + anno['bbox'][2]) / anno['w'], 8)
    bbox4 = round((anno['bbox'][1] + anno['bbox'][3]) / anno['h'], 8)
    LabelName = id_map(anno['category_id'])
    list = []
    list.append(anno['ImageID'])
    list.append(LabelName)
    list.append(bbox1)
    list.append(bbox3)
    list.append(bbox2)
    list.append(bbox4)
    list.append(anno['iscrowd'])
    return list

with open('COCO_eval/instances_val2017.json', 'r') as f:
    j = json.load(f)
    annotation = j['annotations']
    images = j['images']
    all_annotation_list = []
    all_image_list = []
    num = 0
    for i in annotation:
        num += 1
        print(num)
        templist = trans_anno(i, images)
        if isinstance(templist[1], int) == False:
            all_annotation_list.append(templist)
            if templist[0]+'.jpg' not in all_image_list:
                all_image_list.append(templist[0]+'.jpg')
                shutil.copyfile(os.path.join('/media/data/ccr/COCO-val2017/', templist[0]+'.jpg'), os.path.join('/media/data/ccr/COCO-4660/', templist[0]+'.jpg'))

    with open('COCO_eval/test.csv', 'w') as csvvv:
        # create the csv writer
        writer = csv.writer(csvvv)
        # write a row to the csv file
        header = ['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax', 'IsGroupOf']
        writer.writerow(header)
        writer.writerows(all_annotation_list)

    print('annotation num:  ', len(all_annotation_list))
    print('image num:  ', len(all_image_list))


=======
import json
import csv
import os
import shutil

# fw=open("COCO_eval/instance_val2017.json","r",encoding='utf-8')

def id_map(label):
    if label == 1:
        return 'person'
    if label == 3:
        return 'car'
    if label == 4:
        return 'motorcycle'
    if label == 5:
        return 'airplane'
    if label == 6:
        return 'bus'
    if label == 7:
        return 'train'
    if label == 8:
        return 'truck'
    if label == 9:
        return 'boat'
    if label == 10:
        return 'traffic_light'
    if label == 16:
        return 'bird'
    if label == 17:
        return 'cat'
    if label == 18:
        return 'dog'
    if label == 19:
        return 'horse'
    if label == 20:
        return 'sheep'
    if label == 22:
        return 'elephant'
    if label == 24:
        return 'zebra'
    if label == 25:
        return 'giraffe'
    if label == 27:
        return 'backpack'
    if label == 31:
        return 'handbag'
    if label == 32:
        return 'tie'
    if label == 33:
        return 'suitcase'
    if label == 41:
        return 'skateboard'
    if label == 42:
        return 'surfboard'
    if label == 43:
        return 'tennis_racket'
    if label == 44:
        return 'bottle'
    if label == 49:
        return 'knife'
    if label == 50:
        return 'spoon'
    if label == 51:
        return 'bowl'
    if label == 53:
        return 'apple'
    if label == 54:
        return 'sandwich'
    if label == 55:
        return 'orange'
    if label == 56:
        return 'broccoli'
    if label == 57:
        return 'carrot'
    if label == 59:
        return 'pizza'
    if label == 61:
        return 'cake'
    if label == 63:
        return 'couch'
    if label == 70:
        return 'toilet'
    if label == 73:
        return 'laptop'
    if label == 84:
        return 'book'
    if label == 85:
        return 'clock'
    if label == 86:
        return 'vase'
    if label == 60:
        return 'donut'
    if label == 47:
        return 'cup'
    if label == 77:
        return 'cell_phone'
    if label == 76:
        return 'keyboard'
    if label == 21:
        return 'cow'
    if label == 32:
        return 'sports_ball'
    else:
        return 1

def trans_anno(anno, images):
    # del anno['segmentation']
    # del anno['area']
    # del anno['id']
    for i in images:
        if i['id'] == anno['image_id']:
            anno['ImageID'] = i['file_name'][:-4]
            anno['h'] = i['height']
            anno['w'] = i['width']

    bbox1 = round(anno['bbox'][0] / anno['w'], 8)
    bbox2 = round(anno['bbox'][1] / anno['h'], 8)
    bbox3 = round((anno['bbox'][0] + anno['bbox'][2]) / anno['w'], 8)
    bbox4 = round((anno['bbox'][1] + anno['bbox'][3]) / anno['h'], 8)
    LabelName = id_map(anno['category_id'])
    list = []
    list.append(anno['ImageID'])
    list.append(LabelName)
    list.append(bbox1)
    list.append(bbox3)
    list.append(bbox2)
    list.append(bbox4)
    list.append(anno['iscrowd'])
    return list

with open('COCO_eval/instances_val2017.json', 'r') as f:
    j = json.load(f)
    annotation = j['annotations']
    images = j['images']
    all_annotation_list = []
    all_image_list = []
    num = 0
    for i in annotation:
        num += 1
        print(num)
        templist = trans_anno(i, images)
        if isinstance(templist[1], int) == False:
            all_annotation_list.append(templist)
            if templist[0]+'.jpg' not in all_image_list:
                all_image_list.append(templist[0]+'.jpg')
                shutil.copyfile(os.path.join('/media/data/ccr/COCO-val2017/', templist[0]+'.jpg'), os.path.join('/media/data/ccr/COCO-4660/', templist[0]+'.jpg'))

    with open('COCO_eval/test.csv', 'w') as csvvv:
        # create the csv writer
        writer = csv.writer(csvvv)
        # write a row to the csv file
        header = ['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax', 'IsGroupOf']
        writer.writerow(header)
        writer.writerows(all_annotation_list)

    print('annotation num:  ', len(all_annotation_list))
    print('image num:  ', len(all_image_list))


>>>>>>> be0df11a27051c4085a72cd9a58dcd1fea1d076a
