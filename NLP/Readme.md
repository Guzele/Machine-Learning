Project to identify west slavic languages (Czech, Slovak and Polish) and Hungarian.

Structure of project:
  data/source/.. - dataset

  data/cleaned... - cleaned dataset
to clean dataset run cleaner.py

  data/samples/... - full computed dataset in form of matrix
  data/train_test/... - train and test sets
to create them run data_extractor.py

  dnn.py - used to train and test model

  model.json
  model.h5  - saved model

Dataset created by https://github.com/AnnaAK and used sources: http://wiki.dbpedia.org/Downloads2014?v=jd3, http://ftp.acc.umu.se/mirror/wikimedia.org/dumps/skwiki/20180101/, http://zlatyfond.sme.sk/autori


Total time used for training: 0:23:02.020427

acc: 97.67%

             precision    recall  f1-score   support

         cs       0.95      0.98      0.96     62576
         pl       1.00      1.00      1.00     50062
         sk       1.00      1.00      1.00     50130
         hu       0.96      0.91      0.94     37232

avg / total       0.98      0.98      0.98    200000
