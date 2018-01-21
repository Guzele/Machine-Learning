Simple recommender system, which is written without usage of any library.

to run type python main.py

Structure of project:

main.py - to run
data.py - loads data and normalizes it. Also functions to create new.txt

new.txt - dataset from https://labrosa.ee.columbia.edu/millionsong/tasteprofile but cut to have only 1000 top users

learn.py - training of model and making predictions

metrics.py - RMSE, MAE, nDCG metrics
