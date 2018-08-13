import os, csv
import random, pandas

batch_size=64
filepath = os.path.join(os.getcwd(), "data_train")
train_data_file = os.path.join(filepath, "train_data.csv")
n = sum(1 for line in open(train_data_file)) - 1 #number of records in file (excludes header)
skip = sorted(random.sample(range(1,n+1),n-batch_size)) #the 0-indexed header will not be included in the skip list
event_records = pandas.read_csv(train_data_file, skiprows=skip, engine='python')
print(event_records)
