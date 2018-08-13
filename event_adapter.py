import os, csv

filepath = os.path.join(os.getcwd(), "data_train")
train_data_file = os.path.join(filepath, "0809_train.csv")
with open (train_data_file, "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # print (row)
        pass
