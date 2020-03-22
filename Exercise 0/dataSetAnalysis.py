import arff, pandas as pd
data = arff.load(open('c:\\git\\MachineLearning2020S\\MachineLearning2020S\\Exercise 0\\communities_data\\communities.arff', 'r'))
print(data['attributes'])
# df = pd.from_dataframe(data)
for key in data.keys:
    print(key)