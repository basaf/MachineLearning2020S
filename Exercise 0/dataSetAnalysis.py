# -*- coding: utf-8 -*-
from arff import load
import pandas as pd
import matplotlib.pyplot as plt
import configuration as cfg
import os

# load dataset (.arff) into pandas DataFrame
data = load(open(os.path.join(cfg.default.dataset_1_path, 'communities.arff'), 'r'))
all_attributes = list(i[0] for i in data['attributes'])
communities_data = pd.DataFrame(columns=all_attributes, data=data['data'])

# divide attributes in not_predictive, predictive and goal
not_predictive_attributes = [
    'state',
    'county',
    'community',
    'communityname',
    'fold'
]
goal_attribute = 'ViolentCrimesPerPop'

predictive_attributes = all_attributes.copy()
predictive_attributes.remove(goal_attribute)

for x in not_predictive_attributes:
    predictive_attributes.remove(x)

# plot stats
print(f'communities_data')
print(f'min values \r\n{communities_data.min()}\r\n')
print(f'max values \r\n{communities_data.max()}\r\n')
print(f'mean values \r\n{communities_data.mean()}\r\n')
print(f'relative missing values (%) \r\n{str(communities_data.isnull().sum()/len(communities_data.index)*100)})')
plot = (communities_data[predictive_attributes].isnull().sum()/len(communities_data.index)*100)[communities_data[predictive_attributes].isnull().sum()>0].sort_values(ascending=True).plot(kind='barh', grid=True)
plt.xlabel('missing values (%)')
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.dataset_1_figures_path, 'communities_data_missing_values_predictive.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
plt.close()

# plot and save histograms
hist = communities_data[predictive_attributes].hist(figsize=(27, 36))
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.dataset_1_figures_path, 'communities_data_histogram_predictive.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
plt.close()

hist = communities_data[goal_attribute].hist()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.dataset_1_figures_path, 'communities_data_histogram_goal.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
plt.close()

occupancyTrainingData = pd.read_csv(os.path.join(cfg.default.dataset_2_path, 'datatraining.txt'))
occupancyTrainingData = occupancyTrainingData.set_index('date')
occupancyTrainingData['HumidityRatio g/kg'] = occupancyTrainingData['HumidityRatio']*1000 

# plot stats
print(f'min training values \r\n{occupancyTrainingData.min()}')
print(f'max training values \r\n{occupancyTrainingData.max()}')
print(f'mean training values \r\n{occupancyTrainingData.mean()}')
print(f'relative missing values (%) \r\n{str(occupancyTrainingData.isnull().sum()/len(occupancyTrainingData.index)*100)})')

occupancyTrainingData['Temperature'].plot()
plt.xticks(rotation=45)
plt.tight_layout()
plt.xlabel("")
plt.ylabel("Temperature [°C]")
plt.savefig(os.path.join(cfg.default.dataset_2_figures_path, 'ds_2_temperature.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})

# plot and save histograms
hist = occupancyTrainingData.hist()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.dataset_2_figures_path, 'occupancy_data_training_histogram.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
plt.close()

occupancy_test_data = pd.read_csv(cfg.default.dataset_2_path + '\\datatest.txt')
occupancy_test_data = occupancy_test_data.set_index('date')
occupancy_test_data['HumidityRatio g/kg'] = occupancy_test_data['HumidityRatio']*1000 

print(f'min test values \r\n{occupancy_test_data.min()}')
print(f'max test values \r\n{occupancy_test_data.max()}')
print(f'mean test values \r\n{occupancy_test_data.mean()}')
print(f'relative missing values (%) \r\n{str(occupancy_test_data.isnull().sum()/len(occupancy_test_data.index)*100)})')

# plot and save histograms
hist = occupancy_test_data.hist()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.dataset_2_figures_path, 'occupancy_data_test_histogram.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
plt.close()

occupancy_test2_data = pd.read_csv(cfg.default.dataset_2_path + '\\datatest2.txt')
occupancy_test2_data = occupancy_test2_data.set_index('date')
occupancy_test2_data['HumidityRatio g/kg'] = occupancy_test2_data['HumidityRatio']*1000 

print(f'min test2 values \r\n{occupancy_test2_data.min()}')
print(f'max test2 values \r\n{occupancy_test2_data.max()}')
print(f'mean test2 values \r\n{occupancy_test2_data.mean()}')
print(f'relative missing values (%) \r\n{str(occupancy_test2_data.isnull().sum()/len(occupancy_test2_data.index)*100)})')

# plot and save histograms
hist = occupancy_test2_data.hist()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.dataset_2_figures_path, 'occupancy_data_test2_histogram.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
plt.close()
