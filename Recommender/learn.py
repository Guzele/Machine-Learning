import math

#cosine similarity
def _user_similarity(user1, user2):
    commons = list(set(user1.keys()) & set(user2.keys()))
    common_sum = sum([user1[i] * user2[i] for i in commons])

    values1 = list(user1.values())
    values2 = list(user2.values())
    dist1 = sum([x ** 2 for x in list(user1.values())])
    dist2 = sum([x ** 2 for x in list(user2.values())])
    dist = math.sqrt(dist1 * dist2)

    return common_sum /dist if dist != 0 else 0

def user_similarity(data):
    print ("Training model ...")
    user_sim = {}
    for user1_name in data:
       user_sim[user1_name]= {}
       user1 = data[user1_name]
       for user2_name in data: 
          user2 = data[user2_name]
          user_sim[user1_name][user2_name] = _user_similarity(user1, user2)
    return user_sim

def _predict(data, user_sim, user, song):
     sum = 0
     dist = 0

     if not user in user_sim:
        return 0
     
     for user2 in user_sim:
        if not (song in data[user2]):
           continue
        sum += user_sim[user][user2] * data[user2][song]
        dist += user_sim[user][user2]
     return sum /dist if dist != 0 else 0

def predict(data, user_sim, test_data):
     prediction = {}
     for user in test_data:
        prediction[user]= {}
        for song in test_data[user]:
          #print (type(_predict(data, user_sim, user, song)))
          prediction[user][song] = _predict(data, user_sim, user, song)
     return prediction

    





  
