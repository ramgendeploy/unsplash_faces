import numpy as np
import pandas as pd
import glob
import os
from io import BytesIO
import requests
from PIL import Image
import cv2
from IPython.display import display
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from time import sleep
from random import random
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import argparse

def cropface(image, box, fill=.5, ratios=(1,1)):
    shape = image.shape
    if len(shape) > 2 :
        h_img,w_img,c_img = shape
    else:
        h_img,w_img = shape

    Ry, Rx = ratios
    x,y,w,h = box
    

    new_y,new_x = Ry*y, Rx*x
    y_fill = max(0, new_y-h*fill)
    x_fill = max(0, new_x-w*fill)

    new_h, new_w = Ry*(h+y), Rx*(w+x)
    
    h_fill = min(h_img, new_h+h*fill)
    w_fill = min(w_img, new_w+w*fill)
    
    return image[int(y_fill):int(h_fill),
               int(x_fill):int(w_fill)]


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default='./unsplash_faces')
    
    opt = parser.parse_args()
    return opt
  
if __name__ == '__main__':
  opt:dict = get_opt()
  Path(opt.output).mkdir(exist_ok=True, parents=True)

  print('Loading photos df')
  photos_df=pd.read_csv(opt.source, sep=';', header=0)
  print('Finish photos df')
  
  # selected_urls_c = photos_df[['photo_id','photo_url','photo_image_url', 'ai_description']]
  # img_list=selected_urls_c['photo_image_url'].tolist()
  
  for j, r in photos_df.iterrows():
    
    cur_cords = eval(r['face_box_cords'])
    cur_img = r['photo_image_url']
    # print(cur_img,cur_cords)
    image_bytes = requests.get(cur_img.split('?')[0])

    image_bytes = image_bytes.content
    image_stream = BytesIO(image_bytes)
    img_open = np.array(Image.open(image_stream))
    
    name = cur_img.split('/')[-1]
    for i, cords in enumerate(cur_cords):
      print(cords)
      cur_crop = cropface(img_open, cords, fill=0, ratios=(1,1))
      try:
        cv2.imwrite(os.path.join(opt.output,f'{name}_{i}.jpg'), cur_crop[:,:,::-1])
      except:
        print(f'Error with {cur_img}')
        
    print(f'{j}/{len(photos_df)}')