import math

from learn import predict

def RMSE (test_data, prediction):
     N = 0
     sum = 0 
     for user in test_data:
        for song in test_data[user]:
          sum += (test_data[user][song] - prediction[user][song]) ** 2
          N += 1

     return math.sqrt(sum / N)      

def MAE (test_data, prediction):
     N = 0
     sum = 0 
     for user in test_data:
        for song in test_data[user]:
          sum += (test_data[user][song] - prediction[user][song])
          N += 1

     return abs(sum / N)
def DCG (data):
   sum = 0
   N = 0 
   for user in data:
     user_data = data[user]
     sorted_user_data = sorted(user_data.items(), key=lambda kv: kv[1], reverse=True)

     index = 1
     for (song, rate) in  sorted_user_data:
         log = math.log2(float(index))
         maximum = max(1.0, log)
         sum += rate / maximum

         index += 1
         N += 1
   return sum / N if N != 0 else 0
       
    

def NDCG (test_data, prediction):
   return DCG(prediction) / DCG(test_data)

