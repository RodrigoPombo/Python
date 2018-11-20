
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from matplotlib import pyplot


data = pd.read_csv('D:\\Profile\\NB23980\\Downloads\\ml-100k\\u.data.psv',sep='\t', names= ["user_id","item_id","rating","timestamp"])
data['item_id']=pd.to_numeric(data['item_id'])

data1 = pd.read_csv('D:\\Profile\\NB23980\\Downloads\\ml-100k\\u.item.psv',sep='|', names= ["movie_id","title","date","video_date","URL","unknown","action","adventure","animation","children","comedy","crime","documentary","drama","fantasy","film-noir","horror","musical","mystery","romance","sci","thriller","war","western"], encoding = "ISO-8859-1")

data2= data.groupby(by=["item_id"])["rating"].mean()

data2['item_id']= data2.index

data2['item_id']=pd.to_numeric(data2['item_id'])

data2=data2.to_frame()

data1 = data1.set_index('movie_id')

#join index to index
my_result=data2.join(data1, how='right')

ls = ['rating','unknown','action','adventure','animation','children','comedy','crime','documentary','drama','fantasy','film-noir','horror','musical','mystery','romance','sci','thriller','war','western']

my_result=my_result.loc[:, ls]

pca = PCA(n_components=10)
pca.fit(my_result)
var = (pca.explained_variance_ratio_)

com = 0 
ls = list()

for i in range(len(var)):
    com  = com + var[i]
    ls.append(com)
    print(com)

pyplot.plot(ls)

#não valia implementar o kmeans 

ls1 = ['user_id','item_id','rating']
my_data = data.loc[:,ls1]
my_data = my_data.set_index('item_id')
data1 = pd.read_csv('D:\\Profile\\NB23980\\Downloads\\ml-100k\\u.item.psv',sep='|', names= ["movie_id","title","date","video_date","URL","unknown","action","adventure","animation","children","comedy","crime","documentary","drama","fantasy","film-noir","horror","musical","mystery","romance","sci","thriller","war","western"], encoding = "ISO-8859-1")
data1 = data1.set_index('movie_id')
my_result1=my_data.join(data1, how='left')


movieRatings = my_result1.pivot_table(index=['user_id'],columns=['title'],values='rating',aggfunc=lambda x: x.sum(),fill_value=0)


tmpmovieratings = pd.DataFrame(index = movieRatings.columns , columns = movieRatings.columns)

from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

#for i in range(0,len(tmpmovieratings.columns)-1600) :
#    for j in range(0,len(tmpmovieratings.columns)-1600) :
#        tmpmovieratings.ix[i,j] = pearsonr(movieRatings.ix[:,i],movieRatings.ix[:,j])

#for i in range(0,1) :
#    for j in range(0,10) :
#        tmpmovieratings.ix[i,j] = pearsonr(movieRatings.ix[:,i],movieRatings.ix[:,j])

movieRatings1 = movieRatings.ix[:500,:50]
tmpmovieratings = pd.DataFrame(index = movieRatings1.columns , columns = movieRatings1.columns)

for i in range(0,len(tmpmovieratings.columns)) :
    for j in range(0,len(tmpmovieratings.columns)) :
        tmpmovieratings.ix[i,j] = pearsonr(movieRatings1.ix[:,i],movieRatings1.ix[:,j])

similar_movies = pd.DataFrame(index=tmpmovieratings.columns,columns=range(1,7))
for i in range(0,len(tmpmovieratings.columns)): 
    similar_movies.ix[i,:6] = tmpmovieratings.ix[0:,i].sort_values(ascending=False)[:6].index

similar_movies = similar_movies.iloc[:,1:]

movies = pd.read_csv('D:\\Profile\\NB23980\\Downloads\\ml-100k\\u.item.psv',sep='|', names= ["movie_id","title","date","video_date","URL","unknown","action","adventure","animation","children","comedy","crime","documentary","drama","fantasy","film-noir","horror","musical","mystery","romance","sci","thriller","war","western"], encoding = "ISO-8859-1")
data = pd.read_csv('D:\\Profile\\NB23980\\Downloads\\ml-100k\\u.data.psv',sep='\t', names= ["user_id","item_id","rating","timestamp"])

data = data.set_index('item_id')
movies = movies.set_index('movie_id')
my_data= data.join(movies, how='left') 
my_data = my_data[my_data['rating'] >= 4]
my_data = my_data.set_index('title')
#reccomended movies to different users
my_data = my_data.join(similar_movies, how='left')

movieRatings.head()

    
import easygui
my_var = easygui.enterbox("Which is the ID of the user that you want?")

my_var=int(my_var)

my_data1=my_data[my_data['user_id']==my_var][2]
my_data1.sort_values(ascending=False)
print(my_data1.head(1))






