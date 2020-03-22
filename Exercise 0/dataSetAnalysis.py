from arff import load
from pandas import DataFrame
# import sklearn
import matplotlib.pyplot as plt 

data = load(open('c:\\git\\MachineLearning2020S\\MachineLearning2020S\\Exercise 0\\communities_data\\communities.arff', 'r'))
attributes = list(i[0] for i in data['attributes'])

df = DataFrame(columns=attributes, data=data['data'])
# print(df.describe())

histogram = df.hist(figsize=(300, 400))
plt.savefig('communities_data_histogram.png')
print('Fertig')