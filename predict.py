# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure each string is either "ODD"
# or "EVEN" (without the quotes) depending on whether the hexadecimal number in
# the image is odd or even. Take care not to make spelling or case mistakes. Make
# sure that the length of the list returned as output is the same as the number of
# filenames that were given as input. The judge may give unexpected results if this
# convention is not followed.
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from PIL import Image
import os
import pickle
def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dimensions = image.shape
    R = image[0, 0, 0]
    G = image[0, 0, 1]
    B = image[0, 0, 2]
    h, w, c = image.shape
    for i in range(h):
        for j in range(w):
            if R == image[i, j, 0] and G == image[i, j, 1] and B == image[i, j, 2]:
                image[i, j, 0] = 255
                image[i, j, 1] = 255
                image[i, j, 2] = 255
    _, binary = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    image = cv2.dilate(binary, kernel, iterations=1)
    for i in range(h):
        for j in range(w):
            if 255 != image[i, j, 0]:
                image[i, j, 0] = 0
                image[i, j, 1] = 0
                image[i, j, 2] = 0
            if 255 != image[i, j, 1]:
                image[i, j, 0] = 0
                image[i, j, 1] = 0
                image[i, j, 2] = 0
            if 255 != image[i, j, 2]:
                image[i, j, 0] = 0
                image[i, j, 1] = 0
                image[i, j, 2] = 0

    start_x = 450
    end_x = 350
    image = image[:, end_x:start_x]
    return image

def decaptcha( filenames ):
  lt=len(filenames)
  p1 = np.zeros((lt,100,100,3))
  num_train = lt
  for i in range(num_train):
    p1[i] = process_image(filenames[i])
  img_array = np.array(p1, dtype=np.uint8)
  for i, image in enumerate(img_array):
     pil_image = Image.fromarray(image)
     pil_image.save("/content/assignment2/imgn_%d.png"%i)
  data_dir = "/content/assignment2"
  image_size = (100, 100)
  images = []
  for i in range(lt):
    image_path = os.path.join(data_dir, "imgn_" + str(i) + ".png")
    image = Image.open(image_path).convert("L")
    image = np.array(image.resize(image_size)).flatten()
    images.append(image)
  model_file = 'modeldt.pkl'
  with open(model_file, 'rb') as file:
    model = pickle.load(file)
  labels=model.predict(images)
  return labels