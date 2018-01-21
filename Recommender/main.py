import time
import datetime
from data import normalize, load_data, split_data
from learn import user_similarity, predict
from metrics import RMSE, MAE, NDCG

iterations = 5
def cross_validation(data):
     train_data, test_data = split_data(data, 0.25)
     
     #learn
     start_time = time.time()
     user_sim  = user_similarity(train_data)
     finish_time = time.time()
     print ("Learning took " + str(datetime.timedelta(seconds=finish_time - start_time)))
     #prediction
     
     prediction = predict(train_data, user_sim, test_data)

     rmse = RMSE (test_data, prediction)
     print ("RMSE value: " + str(rmse)) 

     mae = MAE (test_data, prediction)
     print ("MAE value: " + str(mae))

     ndcg = NDCG (test_data, prediction)
     print ("NDCG value: " + str(ndcg))

def main():
     data, songs = load_data()
     normalize(data)
   
     for i in range(iterations):
        cross_validation(data)
        print("----------------------------")



if __name__ == "__main__":
    main()

    
