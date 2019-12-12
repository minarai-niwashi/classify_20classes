import glob
import shutil
import csv

file_dir = r"train_data_dir"

with open(r"label_data_dir") as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    train_label = []
    for i in range(50001):
        a = l[i][0]
        if i != 0:
            a = int(a.split("\t")[1])
            train_label.append(a)


b = ["\\aquatic_mammals","\\fish","\\flowers","\\food_containers","\\fruit_and_vegetables","\\household_electrical_devices","\\household_furniture","\\insects","\\large_carnivores","\\large_man-made_outdoor_things","\large_natural_outdoor_scenes","\\large_omnivores_and_herbivores","\\medium_mammals","\\non-insect_invertebrates","\\people","\\reptiles","\\small_mammals","\\trees","\\vehicles_1","\\vehicles_2"]
directry = ["\\for_testgen" , "\\for_validgen"]

file_dir_copy = r"copy_file_dir"
files = glob.glob(file_dir+"\\*.png")
files = sorted(files)


for i,f in enumerate(files):
    for j in range(20):
        if int(train_label[i]) == j:
            if i < 5000:
                file_dir_copy += directry[1]
            else:
                file_dir_copy += directry[0]
            file_dir_copy += b[j]
            shutil.copy(f,file_dir_copy)
            file_dir_copy = r"copy_file_dir"
            break
