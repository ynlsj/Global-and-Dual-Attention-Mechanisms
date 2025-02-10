import os

from pathlib import Path




path= '/root/yolov5/datasets/bdd100k/labels/train'
images_path = '/root/yolov5/datasets/bdd100k/images/train'


for i in Path(images_path).glob('*.jpg'):
    if not os.path.exists(os.path.join(path, i.name.replace('jpg', 'txt'))):
         os.system(f'rm {str(i)}')
         print(f'rm {str(i)}')

# import os

# from pathlib import Path




# path_train= '/root/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
# path_val= '/root/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
# images_path = '/root/yolov5/datasets/voc2012/images/train'
# labels_path = '/root/yolov5/datasets/voc2012/labels/train'

# with open(path_train, 'r') as f:
#     a = f.readlines()
# trainset= [i.split()[0] for i in a]
# print(len(trainset))
# # print(trainset)
# with open(path_val, 'r') as f:
#     a = f.readlines()
# valset= [i.split()[0] for i in a]
# # print(valset)
# print(len(valset))
# for i in Path(images_path).glob('*.jpg'):
# #     if i.name.split('.')[0] in trainset:
# #         os.system(f'mv {str(i)} /root/yolov5/datasets/voc2012/images/train')
#     if  i.name.split('.')[0] in valset:
#          os.system(f'cp {str(i)} /root/yolov5/datasets/voc2012/images/val')
#     else:
#         print(f'error {i.name}')

# for i in Path(labels_path).glob('*.txt'):
# #     if i.name.split('.')[0] in trainset:
# #         os.system(f'mv {str(i)} /root/yolov5/datasets/voc2012/labels/train')
#     if  i.name.split('.')[0] in valset:
#          os.system(f'cp {str(i)} /root/yolov5/datasets/voc2012/labels/val')
#     else:
#         print(f'error {i.name}')