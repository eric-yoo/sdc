#coding = cp949

import os, csv
import random, pandas
from sklearn.preprocessing import LabelEncoder

def load_csv():

    encode_index=['주야(6시,18시 기준)', '오전/오후', '요일', '요일_주야','사망자 피해수준(하 0~2 / 중 3~4 / 상 5~)','부상자 피해수준(하 0~5 / 중 6~18 / 상 19~)',  '발생지시도', '발생지시군구', '발생지시도군구', '사고유형_대분류', '사고유형_중분류', '사고유형',
    '사망자피해수준_사고유형중분류', '부상자피해수준_사고유형중분류', '발생지_사고유형대분류', '발생지_사고유형중분류',
    '법규위반_대분류', '법규위반', '도로형태_대분류', '도로형태', '도로형태(전체)', '도로형태*법규위반',
    '당사자종별_1당_대분류', '당사자종별_1당', '당사자종별_2당_대분류', '당사자종별_2당',  '가해자_피해자', '가해자_도로형태', '가해자_법규위반']

    filepath = os.path.join(os.getcwd(), "data_train")
    train_data_file = os.path.join(filepath, "train.csv")
    df = pandas.read_csv(train_data_file, encoding="utf-8", engine='python')
    # df.dtypes.index
    event_records = df.values

    # print(event_records)
    encoder_list=[]
    for i in encode_index:
        le = LabelEncoder()
        df[i]= le.fit_transform(df[i])
        encoder_list.append(le)
    print(df)
load_csv()
