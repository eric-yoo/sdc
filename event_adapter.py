#coding = cp949

import os, csv
import random, pandas

batch_size=64
filepath = os.path.join(os.getcwd(), "data_train")
train_data_file = os.path.join(filepath, "train_data.csv")
n = sum(1 for line in open(train_data_file, encoding="euc-kr")) - 1 #number of records in file (excludes header)
skip = sorted(random.sample(range(1,n+1),n-batch_size)) #the 0-indexed header will not be included in the skip list
# tmp = pandas.read_csv(train_data_file, encoding="euc-kr")
# n = tmp.shape[0]
df = pandas.read_csv(train_data_file, encoding="euc-kr", skiprows=skip, engine='python')
print(df)

# event_records = df.values
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#
# print(onehot_encoded)
