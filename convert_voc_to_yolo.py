import os
import xml.etree.ElementTree as ET
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm


def download_and_extract(url, dest_path):
    # Ensure the directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

        # Extract the tar file
        with tarfile.open(dest_path) as tar:
            tar.extractall(path=dest_path.parent)
    else:
        print(f"Failed to download the dataset from {url}")


def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_label(path, lb_path, year, image_id, names):
    in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in names and int(obj.find('difficult').text) != 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = names.index(cls)  # class id
            out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + '\n')


def main():
    # Define your classes
    names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # URLs for the dataset
    trainval_url = "http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_trainval.tar"
    test_url = "http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_test.tar"

    # Dataset root directory
    dir = Path('../datasets/VOC')

    # Download and extract datasets
    download_and_extract(trainval_url, dir / 'voc2006_trainval.tar')
    download_and_extract(test_url, dir / 'voc2006_test.tar')

    # Convert
    path = dir / 'VOCdevkit'
    for year, image_set in [('2006', 'trainval'), ('2006', 'test')]:
        imgs_path = dir / 'images' / f'{image_set}{year}'
        lbs_path = dir / 'labels' / f'{image_set}{year}'
        imgs_path.mkdir(exist_ok=True, parents=True)
        lbs_path.mkdir(exist_ok=True, parents=True)

        with open(path / f'VOC{year}/ImageSets/Main/{image_set}.txt') as f:
            image_ids = f.read().strip().split()
        for id in tqdm(image_ids, desc=f'{image_set}{year}'):
            lb_path = (lbs_path / f'{id}.jpg').with_suffix('.txt')  # new label path
            convert_label(path, lb_path, year, id, names)  # convert labels to YOLO format


if __name__ == '__main__':
    main()
