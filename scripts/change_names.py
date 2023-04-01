import cv2
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type = str, required = True)
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument('--token_name', type = str, required = True)
 
args = parser.parse_args()
imgs = glob(f'{args.input_dir}/*')
print(f'Total images: {len(imgs)}')

if not os.path.exists(args.input_dir):
    print("Input Directory doesn't exist! Exiting...")


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok = True)


for idx, p in enumerate(imgs):

    img = cv2.imread(p)
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if idx == 0:
        cv2.imwrite(f'{args.output_dir}/{args.token_name}.jpg', img)
    else:
        cv2.imwrite(f'{args.output_dir}/{args.token_name}({idx}).jpg', img)