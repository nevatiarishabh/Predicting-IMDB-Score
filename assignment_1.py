#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('movie_metadata.csv')

#Finding duplicate movies
duplicates = dataset.duplicated(subset='movie_title',keep=False)
sum(duplicates)

#Removing duplicates
dataset = dataset.drop_duplicates(subset='movie_title',keep='first')

#Null values in each column
nandata=dataset.isnull().sum().to_frame('Number of NaN')

#Fill NaN with median and mode
median=dataset[['num_critic_for_reviews','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','num_user_for_reviews','gross','budget','actor_2_facebook_likes','movie_facebook_likes']].median()
dataset=dataset.fillna(median)
mode=dataset[['color','duration','facenumber_in_poster','language','country','content_rating','title_year','aspect_ratio']].mode().iloc[0]
dataset=dataset.fillna(mode)

#Float to int
dataset[['num_critic_for_reviews','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','facenumber_in_poster','num_user_for_reviews','title_year','actor_2_facebook_likes']]=dataset[['num_critic_for_reviews','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','facenumber_in_poster','num_user_for_reviews','title_year','actor_2_facebook_likes']].astype(int)

#Remove row if 1 NaN is left
dataset=dataset.dropna()

#Drop imdb links columns
dataset=dataset.drop(columns=['movie_imdb_link'])

#Seeing the categories of data
dataset['color'].value_counts()
dataset['director_name'].value_counts()
dataset['country'].value_counts()
dataset['language'].value_counts()
dataset['actor_1_name'].value_counts()
dataset['actor_2_name'].value_counts()
dataset['actor_3_name'].value_counts()
dataset['content_rating'].value_counts()
dataset['title_year'].value_counts()

#Making 3 categories of country
value_counts=dataset["country"].value_counts()
vals = value_counts[:2].index
dataset['country'] = dataset.country.where(dataset.country.isin(vals), 'other')
dataset["country"].value_counts()

#Removing unnecessay columns
dataset=dataset.drop(columns=['director_name'])
dataset=dataset.drop(columns=['actor_1_name'])
dataset=dataset.drop(columns=['actor_2_name'])
dataset=dataset.drop(columns=['actor_3_name'])
dataset=dataset.drop(columns=['plot_keywords'])
dataset=dataset.drop(columns=['color'])
dataset=dataset.drop(columns=['language'])
dataset=dataset.drop(columns=['title_year'])
dataset=dataset.drop(columns=['movie_title'])

#Splitting different genres
dataset = dataset.join(dataset.pop('genres').str.get_dummies('|'))

#Combining actors facebook likes
dataset['actors_facebook_likes']=dataset['actor_1_facebook_likes']+dataset['actor_2_facebook_likes']+dataset['actor_3_facebook_likes']
dataset=dataset.drop(columns=['actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes'])

#Making a profit column 
dataset['profit']=dataset['gross']-dataset['budget']

#Making 4 categories of content_ratings
value_counts2=dataset["content_rating"].value_counts()
vals2 = value_counts2[:3].index
dataset['content_rating'] = dataset.content_rating.where(dataset.content_rating.isin(vals2), 'other')
dataset["content_rating"].value_counts()
 
#Handling the categorical data
dataset = pd.get_dummies(data = dataset, columns = ['country'] , prefix = ['country'] , drop_first = True)
dataset = pd.get_dummies(data = dataset, columns = ['content_rating'] , prefix = ['content_rating'] , drop_first = True)

# categorize the imdb values in the range of 0-4,4-6,6-8 and 8-10 to mark them as the bad,average,good and excellent movies respectively
dataset["imdb_binned_score"]=pd.cut(dataset['imdb_score'], bins=[0,4,6,8,10], right=True, labels=False)+1

#Dropping the imdb_score column as it is being replaced with the imdb_binned_score values 
dataset.drop('imdb_score',axis=1,inplace=True)

#Making dependent and independent variables
X=pd.DataFrame(columns=['num_critic_for_reviews','duration','director_facebook_likes','gross','num_voted_users','cast_total_facebook_likes','facenumber_in_poster','num_user_for_reviews','country_USA','country_other','content_rating_PG-13','content_rating_R','content_rating_other','budget','aspect_ratio','movie_facebook_likes','profit','Action','Adventure','Animation','Biography','Comedy','Crime','Documentary','Drama','Family','Fantasy','Film-Noir','History','Horror','Music','Musical','Mystery','News','Romance','Sci-Fi','Short','Sport','Thriller','War','Western'],data=dataset)
y=pd.DataFrame(columns=['imdb_binned_score'],data=dataset)

#Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=100)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Random Forest Classsification
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)

#Prediction and confusion matrix
from sklearn import metrics
rfcpred = rfc.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, rfcpred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, rfcpred))