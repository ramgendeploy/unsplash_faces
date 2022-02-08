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

def face_processing(image_url,
                  haarcascade_frontalface_path='./haarcascade_frontalface_alt_tree.xml', 
                  FacemarkLBF_path='./lbfmodel.yaml', 
                  save_path='./face_box/',
                  width=512):
  crop_faces_l=[]

  detector = cv2.CascadeClassifier(haarcascade_frontalface_path)

  if width:
    image_url += f'?width={width}'
    
  image_bytes = requests.get(image_url)
  status = image_bytes.status_code
  
  image_bytes = image_bytes.content
  image_stream = BytesIO(image_bytes)
  img_open = Image.open(image_stream)

  image_draw = np.array(img_open).copy()

  image_gray = cv2.cvtColor(np.array(img_open), cv2.COLOR_RGB2GRAY)
  faces = detector.detectMultiScale(image_gray)

  # face landmarks cropping

  for cords in faces:
    cv2.rectangle(image_draw,
                  (cords[0],cords[1]),
                  (cords[0]+cords[2],cords[1]+cords[3]),
                  (0,255,0),2)

  # for face_box in faces:
  #   crop_faces_l.append(cropface(np.array(img_open).copy(), face_box, fill=50))

  if len(faces) > 0:
    cv2.imwrite(f"{os.path.join(save_path, image_url.split('/')[-1].split('?')[0])}.jpg", image_draw[:,:,::-1])
    
    result_dict = {
        'photo_image_url': [],
        'face_boxs': []
    }
    
    result_dict['photo_image_url'].append(image_url)
    result_dict['face_boxs'].append(faces)

    results_facebox = pd.DataFrame.from_dict(result_dict)
    results_facebox.to_csv(f"{os.path.join(save_path, image_url.split('/')[-1].split('?')[0])}.csv", index=False)
  
  return faces, image_draw, crop_faces_l, status


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', type=int, required=True)


    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
  
  opt:dict = get_opt()
  print('Loading photos df')
  photos_df=pd.read_csv('./photos.tsv000', sep='\t', header=0)
  print('Finish photos df')
  
  selected_urls_c = photos_df[['photo_id','photo_url','photo_image_url', 'ai_description']]
  img_list=selected_urls_c['photo_image_url'].tolist()
  print(img_list[:5])
  

  save_path='./face_box/'
  img_width=512

  Path(save_path).mkdir(exist_ok=True, parents=True)

  # --------
  # Simple progress indicator callback function
  def save_and_print(future):
      global lock, tasks_total, tasks_completed
      # obtain the lock
      with lock:
          # update the counter
          tasks_completed += 1
          # report progress
          
          ##TODO: if its 200 save that to a processed folder to not do it again
          print(f'{future.result()}:{tasks_completed}/{tasks_total} completed, {tasks_total-tasks_completed} remain.')
          

  # mock test that works for moment
  def get_faces_th(url):  
    get_face_cord = face_processing(url, width=img_width)
    return get_face_cord[-1]

  lock = Lock()
  tasks_total = len(img_list)
  tasks_completed = 0

  with ThreadPoolExecutor(max_workers=opt.w) as executor:

      print('Submit to Threads')
      futures = [executor.submit(get_faces_th, img) for img in img_list]
      print('Futures callbacks')
      for future in futures:
          future.add_done_callback(save_and_print)

  print('Done!')