import numpy as np
import pandas as pd
from lightfm import LightFM
from pymongo import MongoClient
import pymysql
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix


def dcg_at_k(r,k):
    r = r[:k]
    dcg = np.sum(r/np.log2(np.arange(2, len(r) +2)))
    return dcg

def ndcg_at_k(r,k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    return dcg_at_k(r,k)/dcg_max

# compute average ndcg for all users
def evaluate_prediction(predictions):
    '''
    Return the average ndcg for each users
    args:
        predictions: np.array user-item predictions
    returns:
        ndcg: float, computed NDCG
    '''
    ndcgs = []
    # iterate
    for target_user in np.unique(val_userId):
        # get movie ids and ratings associated with the target user.
        target_val_movie_ids = val_itemId[val_userId == target_user] 
        target_val_ratings = val_rating[val_userId == target_user] 

        # compute ndcg for this user
        ndcg = ndcg_at_k(target_val_ratings[np.argsort(-predictions[val_userId == target_user])], k=30)
        ndcgs.append(ndcg)
    ndcg = np.mean(ndcgs)
    return ndcg

## 取得資料並做轉換
connection = MongoClient(host='localhost', port=27017)
dbs = connection['user']
col = dbs['userclick']
data = [i for i in col.find()]

means_data = []
for i in data:
    # print(i['_id'])
    mean = 0
    for j,k in i['click'].items():
        # print([i['_id'],j,k])
        mean+=k
        means_data.append([i['_id'],j,k])

# 抓取每個使用者的最高點級數
us = {i[0]:0 for i in means_data}
# print(us)
for i in us:
    # print(i)
    for j in means_data:
        if j[0] == i:
            if us[i] < j[-1]:
                us[i] = j[-1]

for k,r in us.items():
    for i in means_data:
        if i[0] == k:
            mean = i[-1]/r
            i.append(mean)
# print(means_data)


df = pd.DataFrame(means_data, columns=['userId','itemId','clickNo','userMean'])
df1 = pd.read_csv('./new_data.csv')
df1=df1[['userId','itemId','click','timestamp']]
df1['itemId'] = pd.to_numeric(df1.itemId, errors='coerce')
df['itemId'] = df['itemId'].astype('int64')
df['timestamp'] = df1['timestamp']
df2 = df.copy()
df2 = df2.sort_values('timestamp')


user_encorder = LabelEncoder()
item_encorder = LabelEncoder()

user_id = user_encorder.fit_transform(df2.userId)
item_id = item_encorder.fit_transform(df2.itemId)
num_train = int(len(user_id)*0.8)

train_userId = user_id[:num_train]
train_itemId = item_id[:num_train]
train_rating = df2.userMean.values[:num_train]

val_userId = user_id[num_train:]
val_itemId = item_id[num_train:]
val_rating = df2.userMean.values[num_train:]

user2item = np.zeros([user_id.max()+1,item_id.max()+1])
user2item[train_userId,train_itemId]=train_rating
user2item = csr_matrix(user2item)

## TF-IDF

connInfo = {
    'host':'localhost',
    'port':3306,
    'user':'TFI101',
    'passwd':'tfi101',
    'charset':'utf8mb4',
    'db':'shopdb'
}

conn = pymysql.connect(**connInfo)
cursor = conn.cursor()
cursor.execute("select itemname, itemid, cate, tags from item ")
conn.commit()
datas = cursor.fetchall()
cursor.close()
conn.close()

feature_df = pd.DataFrame(datas,columns=['name','itemId','cate','tags'])
feature_df['encId'] = item_encorder.transform(feature_df.itemId)
feature_df.loc[feature_df.itemId.isin(df2.itemId.unique())]
item_feature = feature_df.apply(lambda x:x['cate'].lower()+' '+x['tags'].lower(), axis=1).values
item_feature = [it.split(' ') for it in item_feature]


counter = Counter(np.hstack(item_feature))
word2index = {word:idx for idx, (word, count) in enumerate(sorted(counter.items(),key=lambda x:x[1], reverse=True))}

count_matrix = np.zeros([len(item_feature),len(np.unique(np.hstack(item_feature)))], dtype=np.int32)
for idx, ot in enumerate(item_feature):
    for word in ot:
        if word in word2index:
            count_matrix[idx][word2index[word]] += 1

transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(count_matrix)
tfidf_matrix = tfidf_matrix.toarray()


# built sparse matrix for tfidf features 建造商品特徵矩陣()
item_meta_ids = feature_df.encId.values
# item_meta_ids.shape
item_features = np.zeros([item_id.max()+1,len(np.unique(np.hstack(item_feature)))])
# item_features.shape
item_features[item_meta_ids,:] = tfidf_matrix
# item_features
item_features = csr_matrix(item_features) #轉成稀疏矩陣

# training
# ITEM_ALPHA = 1e-6
# model = LightFM(no_components=32, loss='warp', item_alpha=ITEM_ALPHA)
#no_components=200 矩陣分解的k值 loss損失函數
model = LightFM(no_components=32, loss='warp')
model.fit(interactions=user2item, epochs=2, item_features = item_features)

predictions = model.predict(user_ids= val_userId, item_ids= val_itemId, item_features = item_features)
print(evaluate_prediction(predictions))

def recommend(model, user, top_N):
    predictions = model.predict(user_ids = user, item_ids=np.unique(item_id), item_features=item_features)
    return item_encorder.inverse_transform(np.argsort(-predictions))[:top_N]
    
conn = pymysql.connect(**connInfo)
cursor = conn.cursor()
cursor.execute("use {}".format('SHOPDB'))
cursor.execute("""CREATE TABLE IF NOT EXISTS lightFM_test (
    `userId` INT NOT NULL,
    `item1` VARCHAR(100) NOT NULL,
    `item2` VARCHAR(100) NOT NULL,
    `item3` VARCHAR(100) NOT NULL,
    `item4` VARCHAR(100) NOT NULL,
    `item5` VARCHAR(100) NOT NULL,
    `item6` VARCHAR(100) NOT NULL,
    `item7` VARCHAR(100) NOT NULL,
    `item8` VARCHAR(100) NOT NULL,
    `item9` VARCHAR(100) NOT NULL,
    `item10` VARCHAR(100) NOT NULL);""")


cursor.execute("select userId from lightFM_test ")
conn.commit()
existUser =[i[0] for i in cursor.fetchall()]
# print(existUser)

updatesql = """UPDATE lightFM_test SET item1 = {},item2 = {},item3 = {},item4 = {}, item5 = {},item6 = {},item7 = {},item8 = {},item9 = {},
               item10 = {}  WHERE userId={};"""
insert_commit = """INSERT INTO lightFM_test (userId, item1, item2, item3, item4, item5, item6, item7, item8, item9,item10)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

# result = []
for i in range(max(user_id+1)):
    user = user_encorder.inverse_transform([i]).item()
    # print(type(user))
    comm = recommend(model, i, 10).tolist()
    if user in existUser:
        cursor.execute(updatesql.format(comm[0],comm[1],comm[2],comm[3],comm[4],comm[5],comm[6],comm[7],comm[8],comm[9],user))
        conn.commit()
        print(f"Update {user} into {comm}")

    else:
        user = user_encorder.inverse_transform([i]).tolist()
        # print( type(user),type(i))
        cursor.execute(insert_commit, user+comm)
        conn.commit()
        print(f"Insert {user} into {comm}")
cursor.close()
conn.close()