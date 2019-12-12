#load model
from keras.models import load_model

model = load_model("model_earlystopping.h5")

#predict classification
import numpy as np
from PIL import Image
import csv
import glob
import os

img_width = 150
img_height = 150
image_dir = r"test_data_dir"

csv_file_path = r"predict_data"

print("convert into numpy")
files = glob.glob(image_dir + r"\*.png")
file_name = []

for num, f in enumerate(files):
    probability = []
    file_name.append(os.path.split(f)[1])
    
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((img_width, img_height))
    img_data = np.asarray(img)
    img_data = img_data/255.0
    img_data = np.array([img_data])

    prob = model.predict(img_data)
    
    probability.append(file_name[num])
    for i in range(20):
        probability.append(prob[0][i])

    with open(csv_file_path, "a", newline="") as f:
        writer = csv.writer(f)    
        writer.writerow(probability)
    
    del writer, prob
    del img, img_data,probability