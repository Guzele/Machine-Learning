Project to identify west slavic languages (Czech, Slovak and Polish) and Hungarian.

Structure of project:
data/source/.. - dataset

data/cleaned... - cleaned dataset
to clean dataset run cleaner.py

data/samples/... - full computed dataset in form of matrix
data/train_test/... - train and test sets
to create them run data_extractor.py

dnn.py - used to train and test model

Dataset created by https://github.com/AnnaAK.

acc: 97.70%

             precision    recall  f1-score   support

         cs       0.95      0.98      0.97     62576
         hu       1.00      1.00      1.00     50062
         pl       1.00      1.00      1.00     50130
         sk       0.96      0.92      0.94     37232

