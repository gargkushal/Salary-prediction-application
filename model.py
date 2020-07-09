import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from word2number import w2n
import pickle

df=pd.read_csv('hiring.csv')
df['experience'].fillna('zero',inplace=True)
import math
m=df['test_score(out of 10)'].mean()

q=math.floor(m)
df['test_score(out of 10)'].fillna(q,inplace=True)
df.experience = df.experience.apply(w2n.word_to_num)
reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])
pickle.dump(reg, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
