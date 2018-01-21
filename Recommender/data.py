import random

# dataset from https://labrosa.ee.columbia.edu/millionsong/tasteprofile
fileName = 'train_triplets.txt' 
#dataset with only top users
destName = 'new.txt'

def _get_from_list(list):
    data = {}
    for line in list:
       user, song, count = line
       if user not in data:
          data[user] = {}
       data[user][song] = count
    return data

# data in form of data[user][song] = count
def split_data(data, test_ratio=0.25):
      print ("Spliting dataset in ratio: " + str(test_ratio))

      data_list = []
      for user in data:
         for song in data[user]:
            item =  (user, song, data[user][song])
            data_list.append (item)

      random.shuffle(data_list)

      total = len(data_list)
      print ("Total dataset size: " + str(total))
      test_size = round (test_ratio * total)
      train_size = total - test_size
      print ("Training size: " + str(train_size))
      print ("Testing size: " + str(test_size))
      train_list = data_list[:train_size]
      test_list= data_list[train_size:]

      train_data = _get_from_list(train_list)
      test_data  = _get_from_list(test_list)
      return (train_data, test_data)

def normalize(data):
    # Normalize rates
    for user in data:
        max_count = max([data[user][song] for song in data[user]])
        for song in data[user]:
            data[user][song] = round(data[user][song] * 10.0 / max_count ) + 1
    return data

def load_data():
    print ("Loading dataset ...")
    f = open(destName, 'r')
    data = {}
    songs = set()
    for line in f:
       user, song, count = line.split(sep='\t')
       count = int(count)
       songs.add(song)

       if user not in data:
          data[user] = {}

       data[user][song] = count
       
    f.close()
    return (data, songs)

#local functions which is needed only to cut the dataset to keep top users
max_users = 1000
def _load_users():
    f = open(fileName, 'r')
    users = dict()
    for line in f:
       user = line.split(sep='\t')[0]
       if (user == ""):
             continue
       if user in users:
             users[user] += 1          
       else:
             users[user] = 1  
    f.close()
    return users
def _top_users():
    users = _load_users()
    top_users = sorted(users, key=lambda kv: kv[1], reverse=True)[:max_users]
    return top_users

def _create_top_user_dataset():
    top_users = _top_users()
    f = open(fileName, 'r')
    dest = open(destName, 'w')
    for line in f:
       user = line.split(sep='\t')[0]
       if (user in top_users):
          dest.write (line)
          
    f.close()
    dest.close()


