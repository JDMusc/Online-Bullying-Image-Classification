import os
from PIL import Image
from toolz import pipe as p
from xml.dom import minidom
import xml.etree.ElementTree as ET

import pandas as pd

root = 'C:/Users/anamu/OneDrive/Desktop/Biomedical Data Science and Informatics/2019 Spring/CPSC 8810 Deep Learning/Project/Labelled Image/'


def writeLocationsFiles(locations_dir, dest_dir):
    os.makedirs(dest_dir)
    location_fs = [os.path.join(locations_dir, f) 
        for f in os.listdir(locations_dir) 
        if f.endswith('.csv')]
    
    for f in location_fs:
        (clas, base_img_f) = parseLocationsFileName(f)

        clas_dir = os.path.join(dest_dir, clas)
        if not os.path.exists(clas_dir):
            os.makedirs(clas_dir)

        try:
            dest_f = p(base_img_f,
                lambda _: os.path.splitext(_)[0],
                lambda _: _ + '.xml',
                lambda _: os.path.join(clas_dir, _))
        
            writeLocationsFile(f, dest_f)
        except Exception:
            Warning('file ' + f + ' not parsed')


def writeLocationsFile(locations_f, dest_f):
    xmlstr = p(locations_f, 
        toXml, 
        toXmlString)
    
    with open(dest_f, "w") as f:
        f.write(xmlstr)


def toXmlString(xml):
    return p(xml, 
    ET.tostring,
    minidom.parseString,
    lambda _: _.toprettyxml(),
    lambda _: _.replace('<?xml version="1.0" ?>\n', ''))


def toXml(locations_f):
    (clas, img_f_name) = parseLocationsFileName(locations_f)

    ann = createHeader(clas, img_f_name)

    size = createSizeTag(clas, img_f_name)

    ann.append(size)
    
    locations = pd.read_csv(locations_f)
    n_boxes = locations.shape[0]
    for _ in range(0, n_boxes):
        arr = locations.iloc[_, 0:4].get_values().astype(int)
        object = createObjectTag(arr, clas)
        ann.append(object)

    return ann


def createHeader(clas, img_f_name):
    xml_root = ET.Element('annotation')

    folder = ET.SubElement(xml_root, 'folder')
    folder.text = clas

    filename = ET.SubElement(xml_root, 'filename')
    filename.text = os.path.basename(img_f_name)

    path = ET.SubElement(xml_root, 'path')
    path.text = os.path.join(root, clas, img_f_name)

    source = ET.SubElement(xml_root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    segmented = ET.SubElement(xml_root, 'segmented')
    segmented.text = 0

    return xml_root


def createSizeTag(clas, img_f_name):
    full_img_f = os.path.join('image_data', clas, img_f_name)
    img = Image.open(full_img_f)

    size = ET.Element('size')
    width = ET.SubElement(size, 'width')
    width.text = str(img.width)
    height = ET.SubElement(size, 'height')
    height.text = str(img.height)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(img.layers)

    return size


def createObjectTag(arr, c):
    if len(arr) == 0:
        return None
    object = ET.Element('object')
    name = ET.SubElement(object, 'name')

    if c == 'laughing':
        name.text = 'bully'
    else:
        name.text = 'victim'

    pose = ET.SubElement(object, 'pose')
    pose.text = 'Unspecified'

    truncated = ET.SubElement(object, 'truncated')
    truncated.text = "0"

    difficult = ET.SubElement(object, 'difficult')
    difficult.text = "0"

    bndbox = createBoundingBoxTag(arr)
    object.append(bndbox)

    return object


def createBoundingBoxTag(arr):
    bndbox = ET.Element('bndbox')

    def addElement(name, i):
        tag = ET.SubElement(bndbox, name)
        tag.text = str(arr[i])
    
    addElement('xmin', 0)
    addElement('ymin', 1)
    addElement('xmax', 2)
    addElement('ymax', 3)

    return bndbox
    

def parseLocationsFileName(locations_f):
    base_f = os.path.basename(locations_f)
    (clas, img_f_name) = base_f.split('_')
    img_f_name = img_f_name.replace('.csv','')
    return (clas, img_f_name)