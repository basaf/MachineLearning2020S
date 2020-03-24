# -*- coding: utf-8 -*-
from arff import load
import pandas as pd
import matplotlib.pyplot as plt
import configuration as cfg
import os

# load dataset (.arff) into pandas DataFrame
data = load(open(os.path.join(cfg.default.dataset_1_path, 'communities.arff'), 'r'))
all_attributes = list(i[0] for i in data['attributes'])
df = pd.DataFrame(columns=all_attributes, data=data['data'])

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

hist = df[predictive_attributes].hist(figsize=(27, 36))
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.dataset_1_figures_path, 'communities_data_histogram_predictive.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
plt.close()

hist = df[goal_attribute].hist()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.dataset_1_figures_path, 'communities_data_histogram_goal.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
plt.close()

occupancyTrainingData = pd.read_csv(os.path.join(cfg.default.dataset_2_path, 'datatraining.txt'))
occupancyTrainingData = occupancyTrainingData.set_index('date')

# plot stats
print(f'min training values \r\n{occupancyTrainingData.min()}')
print(f'max training values \r\n{occupancyTrainingData.max()}')
print(f'mean training values \r\n{occupancyTrainingData.mean()}')

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

print(f'min test values \r\n{occupancy_test_data.min()}')
print(f'max test values \r\n{occupancy_test_data.max()}')
print(f'mean test values \r\n{occupancy_test_data.mean()}')

# plot and save histograms
hist = occupancy_test_data.hist()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.dataset_2_figures_path, 'occupancy_data_test_histogram.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
plt.close()

occupancy_test2_data = pd.read_csv(cfg.default.dataset_2_path + '\\datatest2.txt')
occupancy_test2_data = occupancy_test2_data.set_index('date')

print(f'min test2 values \r\n{occupancy_test2_data.min()}')
print(f'max test2 values \r\n{occupancy_test2_data.max()}')
print(f'mean test2 values \r\n{occupancy_test2_data.mean()}')

# plot and save histograms
hist = occupancy_test2_data.hist()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.dataset_2_figures_path, 'occupancy_data_test2_histogram.pdf'), format='pdf',
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
plt.close()
