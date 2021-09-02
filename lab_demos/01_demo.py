import deepchem as dc
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='ECFP')
train_dataset, valid_dataset, test_dataset = datasets
print(train_dataset)

train_dataset.w

model = dc.models.MultitaskClassifier(n_tasks=12, n_features=1024, layer_sizes=[1000])

import numpy as np

model.fit(train_dataset, nb_epoch=10)
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
print('training set score:', model.evaluate(train_dataset, [metric], transformers))
print('test set score:', model.evaluate(test_dataset, [metric], transformers))