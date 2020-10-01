import torch
import warnings
warnings.filterwarnings('ignore')
import sys
#import matplotlib.pyplot as plt
#import os.path as op
import pandas as pd
#pd.options.display.max_columns = None
from ast import literal_eval
#from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import torch.nn as nn
import imp
import numpy as np
import os.path as op
#import torch.optim as optim

model=torch.load('best_model.pth')
#model.eval
i=model['module._net.movie_embeddings.weight']
u=model['module._net.user_embeddings.weight']
def convert_int(x):
  try:
      return int(x)
  except:
      return np.nan
def predicter(userid,moviename):
  userembedding=user.loc[user['user_id'] == userid, 'embedding']
  movieembedding=movies_names.loc[movies_names['moviename']==moviename,'embedding']
  x=userembedding
  y=movieembedding
  x=x.to_list()
  y=y.to_list()
  a=np.array(x)
  b=np.array(y)
  return (np.vdot(a,b))
def getmovieId(tmdbids):
  results=[]
  for i in tmdbids:
    a=master_data.loc[master_data['tmdbId'] == i,'movieId']
    if(len(a)!=0):
      a=np.array(a)
      results.append(a)
  newlist=[]
  for i in range(0,len(results)):
    newlist.append(results[i][0])
  return newlist
def estimate(userId,x):
  results=[]
  for index,rows in x.iterrows():
    a=model._net(torch.LongTensor([int(userId)]),torch.LongTensor([rows['movieId']]))
    a=a.detach().numpy()
    results=np.concatenate((a, results), axis=0)
    #results.append(a)
  return results
def dfmovie(movieIds):
    movies=[]
    for i in movieIds:
       #a=master_data.iloc[i]['movieid','moviename','embedding']
        a=master_data.loc[master_data['movieId']==i,'embedding']
        b=master_data.loc[master_data['movieId']==i,'moviename']
        c=master_data.loc[master_data['movieId']==i,'movieId']
        a=a.to_list()
        a=np.array(a)
        b=b.to_list()
        b=np.array(b)
        c=c.to_list()
        c=np.array(c)
        movies.append([c[0],b[0],a[0]])
    movies=pd.DataFrame(movies,columns=['movieId','moviename','embedding'])
    return movies
def pri(userId,x):
  results=[]
  for index,rows in x.iterrows():
    a=predicter(str(userId),rows['moviename'])
    results.append(a)
  return results


def recommend(userId, title):
  if(title in indices.keys()):
    idx = indices[title]
  else:
    print("enter a movie name from the recommender system")
    sys.exit(0)
  tmdbId = id_map.loc[title]['id']
    #print(idx)
  movie_id = id_map.loc[title]['movieId']
    
  similarity_scores = list(enumerate(cosine_sim[int(idx)]))
  similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
  similarity_scores = similarity_scores[1:100]
  movie_indices = [i[0] for i in similarity_scores]
  x=movie_indices
  movieIds=getmovieId(movie_indices)
  all_ratings = pd.read_csv('datasets/u.data', sep='\t',
                          names=["userId", "movieId", "rating", "timestamp"])
  y=all_ratings.loc[all_ratings['userId']==userId,'movieId'].to_list()
  for i in movieIds:
    if(i in y):
      movieIds.remove(i)
  x=dfmovie(movieIds)
  l=pri(userId,x)
  x['estimate']=l
  #x['estimate']=y
  # x['est'] = x['moviename'].apply(lambda x: predicter(userId, indices_map.loc[x]['moviename']))
  x = x.sort_values('estimate', ascending=False)
  return x


master = pd.read_csv('datasets/movies_metadata.csv')
master['genres'] = master['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] 
                                                                       if isinstance(x, list) else [])
small_mdf = pd.read_csv('datasets/links_small.csv')
  #print(small_mdf.head())
small_mdf = small_mdf[small_mdf['tmdbId'].notnull()]['tmdbId'].astype('int')
master['id'] = master['id'].apply(convert_int)
s_master = master[master['id'].isin(small_mdf)]
s_master['tagline'] = s_master['tagline'].fillna('')
s_master['description'] = s_master['overview'] + s_master['tagline']
s_master['description'] = s_master['description'].fillna('')
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(s_master['description'])
tfidf_matrix.shape
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
s_master = master[master['id'].isin(small_mdf)]
list_movies_names = []
list_item_ids = []
with open(op.join('datasets/u.item'), encoding = "ISO-8859-1") as fp:
  for line in fp:
      list_item_ids.append(line.split('|')[0])
      list_movies_names.append(line.split('|')[1])
        
movies_names = pd.DataFrame(list(zip(list_item_ids, list_movies_names)), 
              columns =['movieId', 'moviename']) 
item_embeddings_np = i.data.cpu().numpy()
movies_names['embedding']=item_embeddings_np[1:].tolist()
list_age = []
list_user_ids = []
with open(op.join('datasets/u.user'), encoding = "ISO-8859-1") as fp:
  for line in fp:
      list_user_ids.append(line.split('|')[0])
      list_age.append(line.split('|')[1])
        
user = pd.DataFrame(list(zip(list_user_ids, list_age)), 
               columns =['user_id', 'age']) 
user_embeddings_np = u.data.cpu().numpy()
user['embedding']=user_embeddings_np[1:].tolist()
s_master = s_master.reset_index()
titles = s_master['title']
indices = pd.Series(s_master.index, index=s_master['title'])
id_map = pd.read_csv('datasets/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(s_master[['title', 'id']], on='id').set_index('title')
indices_map = id_map.set_index('id')
links=pd.read_csv('datasets/links_small.csv')
links['tmdbId'] = links['tmdbId'].apply(convert_int)
links=links[links['tmdbId'].notna()].astype('int')
movies_names['movieId'] = movies_names['movieId'].apply(convert_int)
links['movieId'] = links['movieId'].apply(convert_int)
master_data=pd.merge(movies_names,links,on='movieId')
l=recommend(sys.argv[1],sys.argv[2])
l=l.drop(['embedding'],axis=1)
print(l.head(3))




