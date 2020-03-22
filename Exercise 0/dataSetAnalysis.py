from arff import load
from pandas import DataFrame
data = load(open('c:\\git\\MachineLearning2020S\\MachineLearning2020S\\Exercise 0\\communities_data\\communities.arff', 'r'))
attributes = list(i[0] for i in data['attributes'])

df = DataFrame(columns=attributes, data=data['data'])
print(df)