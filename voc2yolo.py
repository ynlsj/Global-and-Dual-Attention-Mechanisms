# -*- coding: utf-8 -*-

from xml.dom import minidom
import os
import glob

lut = {
'pedestrian': 0,  # bdd100k
'other person': 0,
'rider':0,
# 'aeroplane': 0,
# 'bicycle':1,
# 'bird':2,
# 'boat':3,
# 'bottle':4,
# 'bus':5,
# 'car':6,
# 'cat':7,
# 'chair':8,
# 'cow':9,
# 'diningtable':10,
# 'dog':11,
# 'horse':12,
# 'motorbike':13,
# 'person':14,
# 'pottedplant':15,
# 'sheep':16,
# 'sofa':17,
# 'train':18,
# 'tvmonitor':19,
}


def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_xml2yolo(lut):
    for fname in glob.glob("/root/labels_xml/val/*.xml"):
        print(fname)
        xmldoc = minidom.parse(fname)

        fname_out = (fname[:-4] + '.txt')

#         with open(fname_out, "w") as f:

        itemlist = xmldoc.getElementsByTagName('object')
        size = xmldoc.getElementsByTagName('size')[0]
        width = int((size.getElementsByTagName('width')[0]).firstChild.data)
        height = int((size.getElementsByTagName('height')[0]).firstChild.data)

        for item in itemlist:
            # get class label
            classid = (item.getElementsByTagName('name')[0]).firstChild.data
            if classid not in lut:
#                 print(f'error {fname}')
                continue
            label_str = str(lut[classid])
            # else:
            #     label_str = "-1"
            #     print("warning: label '%s' not in look-up table" % classid)

            # get bbox coordinates
            xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
            ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
            xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
            ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
            b = (float(xmin), float(xmax), float(ymin), float(ymax))
            bb = convert_coordinates((width, height), b)
            # print(bb)
            with open(fname_out, "a") as f:
                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')

        print("wrote %s" % fname_out)


def main():
    convert_xml2yolo(lut)


if __name__ == '__main__':
    main()