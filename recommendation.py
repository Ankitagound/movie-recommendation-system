# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import helper
from sklearn.metrics import silhouette_samples, silhouette_score #Silhouette refers to a method of interpretation and validation of consistency within clusters of data. 

#import the dataset
movies=pd.read_csv(r'file:///C:/Users/Ankita/Desktop/Imdb Dataset/movies.csv')
movies.head()
movies.fillna(0, inplace=True)
ratings=pd.read_csv(r"file:///C:/Users/Ankita/Desktop/Imdb Dataset/ratings.csv")
ratings.head()

#to find out how the structure of the dataset works and how many records do we have in each of these tables.
print('The datset contain:', len(ratings),'ratings of:', len(movies),'movies.')

movies.mean(axis=0)   #average for each column
movies.mean(axis=1)   #average for each row

# Function to get the genre ratings
def get_genre_ratings(ratings, movies, genres, column_names):
    genre_ratings = pd.DataFrame()
    for genre in genres:        
        genre_movies = movies[movies['genres'].str.contains(genre) ]
        avg_genre_votes_per_user = ratings[ratings['movieid'].isin(genre_movies['movieid'])].loc[:, ['userid', 'ratings']].groupby(['userid'])['ratings'].mean().round(2)
        genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)
        
    genre_ratings.columns = column_names
    return genre_ratings
# Calculate the average rating of romance and scifi movies
genre_ratings = get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating'])
genre_ratings.head()

'''we will make some Visualization Analysis in order to obtain a good overview 
of the biased dataset and its characteristics.'''

#Defining the scatterplot drawing function
def draw_scatterplot(x_data,x_label,y_data,y_label):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111) #These are subplot grid parameters encoded as a single integer. For example, "111" means "1x1
    plt.xlim(0,5)
    plt.ylim(0,5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x_data,y_data,s=30)
    
    
#plot the scatterplot
draw_scatterplot(genre_ratings['avg_scifi_rating'],'Avg scifi rating',genre_ratings['avg_romance_rating'],'Average romentic rating')

'''as the output is not proper we are going to bias our data so that we can get
ratings from those users that like either romance or science fiction movies.'''

#function to get the biased dataset 
def biased_rating_dataset(genre_ratings,score_limit_1,score_limit_2):
    biased_dataset = genre_ratings[((genre_ratings['avg_romance_rating'] < score_limit_1 - 0.2) & (genre_ratings['avg_scifi_rating'] > score_limit_2))|
            ((genre_ratings['avg_scifi_rating'] <  score_limit_1) &
             (genre_ratings['avg_romance_rating'] > score_limit_2))]
            
    biased_dataset = pd.concat([biased_dataset[:300],genre_ratings[:2]])
    biased_dataset = pd.DataFrame(biased_dataset.to_records()) #.t0_records() used to convert DataFrame to a NumPy record array.
    
    return biased_dataset

            #Bias  the datset
biased_dataset = biased_rating_dataset(genre_ratings,3.2,2.5)

#printing the number of records & the head of the dataset
print("Number of records:",len(biased_dataset))
biased_dataset.head()            

'''Now, we will make some Visualization Analysis in order 
to obtain a good overview of the biased dataset and its characteristics.'''

#Defining the scatterplot drawing function
def draw_scatterplot(x_data,x_label,y_data,y_label):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111) #These are subplot grid parameters encoded as a single integer. For example, "111" means "1x1
    plt.xlim(0,5)
    plt.ylim(0,5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x_data,y_data,s=30)
    
#plot the scatterplot
draw_scatterplot(biased_dataset['avg_scifi_rating'],'Avg scifi rating',biased_dataset['avg_romance_rating'],'Average romentic rating')

'''The biase that we have created previously is perfectly clear now. 
We will take it to the next level by applying K-Means to break down the sample into two distinct groups.'''

#let's turn our dataset into the list
X= biased_dataset [['avg_scifi_rating','avg_romance_rating']].values

#importing k-means

#creating an instance of KMeans to find two cluster
kmeans_1 = KMeans(n_clusters=2)

#using fit predict to predict the dataset
predict=kmeans_1.fit_predict(X)

#Defining the cluster plotting function
def draw_clusters(biased_dataset, predict, cmap='viridis'): #cmap= colormap
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.set_xlabel('avg scifi rating')
    ax.set_ylabel('avg romance rating')
    
    clustered = pd.concat([biased_dataset.reset_index(),pd.DataFrame({'group':predict})], axis=1) #reset_index() = reseting the index value
    plt.scatter(clustered['avg_scifi_rating'],clustered['avg_romance_rating'] ,c=clustered['group'], s=20 , cmap=cmap)

#plot the cluster
draw_clusters(biased_dataset, predict)

'''People that averaged a rating on romance movies of 3 or higher will belong to one group, 
and people who averaged a rating of less than 3 will belong to the other.'''

#Create an instance of KMeans to find three cluster
kmeans_2 = KMeans(n_clusters=3)

#use fit_predict to cluster the dataset
predict_2=kmeans_2.fit_predict(X)

#plot the cluster
draw_clusters(biased_dataset,predict_2)

#People who like scie-fi and romance belong to the yellow group.
#People who like scifi but not romance belong to the green group.
#People who like romance but not sci-fi belong to the purple group.

#create an instance for 4 clusters
kmeans_3=KMeans(n_clusters=4)

#now we will fit_predict the dataset cluster
predict_3=kmeans_3.fit_predict(X)

#plot the data
draw_clusters(biased_dataset,predict_3)

''' we will now choose the right K numbers of clusters
we will choose the elbow method for our clustering

                       Variance between group
        %Variance= _________________________________                                 
                            Total variance
                            
then we will choose the k=3, where the elbow is located.
One of the ways to calculate this error is by:
    
    -First, subtracting the Euclidean distance from each point of each cluster to the centroid of its respective group.
    -Then, squaring this value (to get rid of the negative terms).
    -And finally, adding all those values, to obtain the total error.
    
    
So, now we want to find out the right number of clusters for our dataset. 
To do so, we are going to perform the elbow method for all the possible values of Kl 
which will range between 1 and all the elements of our dataset. 
That way we will consider every possibility within the extreme cases:
    
    If K = 1, there is only one group which all the points belong to.
    If K = all data points, each data point is a separate group.'''
    
    
# Selecting our dataset to study
df = biased_dataset[['avg_scifi_rating','avg_romance_rating']]

# Choose the range of k values to test.
# We added a stride of 5 to improve performance. We don't need to calculate the error for every k value
possible_k_values = range(2, len(X)+1, 5)

# Define function to calculate the clustering errors
def clustering_errors(k, data):
    kmeans = KMeans(n_clusters=k).fit(data)
    predictions = kmeans.predict(data)
    #cluster_centers = kmeans.cluster_centers_
    # errors = [mean_squared_error(row, cluster_centers[cluster]) for row, cluster in zip(data.values, predictions)]
    # return sum(errors)
    silhouette_avg = silhouette_score(data, predictions)
    return silhouette_avg

# Calculate error values for all k values we're interested in
errors_per_k = [clustering_errors(k, X) for k in possible_k_values]

# Plot the each value of K vs. the silhouette score at that value
fig, ax = plt.subplots(figsize=(16, 6))
plt.plot(possible_k_values, errors_per_k)

#Ticks and grid
xticks=np.arange(min(possible_k_values),max(possible_k_values)+1,5.0)
ax.set_xticks(xticks,minor=False)
ax.set_xticks(xticks,minor=True)
ax.xaxis.grid(True,which='both')
yticks=np.arange(round(min(errors_per_k), 2),max(errors_per_k),.05)
ax.set_yticks(yticks, minor=False)
ax.set_yticks(yticks, minor=True)
ax.yaxis.grid(True,which='both')

# Create an instance of KMeans to find seven clusters
kmeans_4 = KMeans(n_clusters=7)

#predict the clustering data
predict_4 = kmeans_4.fit_predict(X)

#ploting the clustering
draw_clusters(biased_dataset,predict_4, cmap="Accent")


'''we have only analyzed romance and science-fiction movies. 
Let us see what happens when adding other genre to our analysis by adding Action movies.'''

# Select our biased dataset and add action genre
biased_dataset_3_genres = get_genre_ratings(ratings,movies,['Romance','Sci-Fi','Action'],['avg_romance_rating','avg_scifi_rating','avg_action_rating'])

# Print the number of records and the head of our dataset
print( "Number of records: ", len(biased_dataset_3_genres))
biased_dataset_3_genres.head()

#replacing the nan value with 0
biased_dataset_3_genres.fillna(0, inplace=True)

#turn the dataset into list
X_with_action = biased_dataset_3_genres[['avg_scifi_rating','avg_romance_rating','avg_action_rating']].values

#create an instance of KMeans to find 7 cluster
kmeans_5 = KMeans(n_clusters=7)

#use fit predict to cluster the data
predict_5 = kmeans_5.fit_predict(X_with_action)
         
#plot
draw_clusters (biased_dataset_3_genres, predict_5)

'''Here, we are still using the x and y axes of the romance and sci-fi ratings. In addition, 
we are plotting the size of the dot to represent the ratings of the action movies 
(the bigger the dot the higher the action rating).
We can see that with the addition of the action genrem the clustering vary significantly. 
The more data that we add to our k-means model, the more similar the preferences of each group would be.
The bad thing is that by plotting with this method we start loosing the ability to visualize correctly 
when analysing three or more dimensions. So, in the next section we will study other plotting method 
to correctlyy visualize clusters of up to five dimensions.'''

'''High Level Clustering'''

'''we are going to take a bigger picture of the dataset and explore how users rate individual movies.
we will subset the dataset by ‘userid’ vs ‘user rating’ as follows.'''

# Merge the two tables then pivot so we have Users X Movies dataframe
ratings_title = pd.merge(ratings, movies[['movieid', 'title']], on='movieid' )
user_movie_ratings = pd.pivot_table(ratings_title, index='userid', columns= 'title', values='ratings')

# Print he number of dimensions and a subset of the dataset
print('dataset dimension:',user_movie_ratings.shape, '\n\nSubset example:')
user_movie_ratings.iloc[:6, :10]

def get_most_rated_movies(user_movie_ratings, max_number_of_movies):
    # 1- Count
    user_movie_ratings = user_movie_ratings.append(user_movie_ratings.count(), ignore_index=True)
    # 2- sort
    user_movie_ratings_sorted = user_movie_ratings.sort_values(len(user_movie_ratings)-1, axis=1, ascending=False)
    user_movie_ratings_sorted = user_movie_ratings_sorted.drop(user_movie_ratings_sorted.tail(1).index)
    # 3- slice
    most_rated_movies = user_movie_ratings_sorted.iloc[:, :max_number_of_movies]
    return most_rated_movies

def get_users_who_rate_the_most(most_rated_movies, max_number_of_movies):
    # Get most voting users
    # 1- Count
    most_rated_movies['counts'] = pd.Series(most_rated_movies.count(axis=1))
    # 2- Sort
    most_rated_movies_users = most_rated_movies.sort_values('counts', ascending=False)
    # 3- Slice
    most_rated_movies_users_selection = most_rated_movies_users.iloc[:max_number_of_movies, :]
    most_rated_movies_users_selection = most_rated_movies_users_selection.drop(['counts'], axis=1)
    
    return most_rated_movies_users_selection

#def get_users_who_rate_the_most(most_rated_movies,n_users):
 #   return most_rated_movies.sort_values(axis=0, by=movie_list, ascending=False).iloc[:n_users,:]
    
'''it is evident that there are a lot of ‘NaN’ values as most of the users have not rated most of the movies. 
This type of datasets with a number that high of ‘null’ values are called ‘sparse’ or ‘low-dense’ datasets.'''
# Define the sorting by rating function
def sort_by_rating_density(user_movie_ratings, n_movies, n_users):
    most_rated_movies = get_most_rated_movies(user_movie_ratings, n_movies)
    most_rated_movies = get_users_who_rate_the_most(most_rated_movies, n_users)
    return most_rated_movies
# choose the number of movies and users and sort
n_movies = 30
n_users = 18
most_rated_movies_users_selection = sort_by_rating_density(user_movie_ratings,n_movies, n_users)
# Print the result
print('dataset dimensions: ', 
      most_rated_movies_users_selection.shape,
      most_rated_movies_users_selection.head())               

# Define the plotting heatmap function
def draw_movies_heatmap(most_rated_movies_users_selection, axis_labels=True):
    
    fig = plt.figure(figsize=(15,4))
    ax = plt.gca()
    
    # Draw heatmap
    heatmap = ax.imshow(most_rated_movies_users_selection,  interpolation='nearest', vmin=0, vmax=5, aspect='auto')
    if axis_labels:
        ax.set_yticks(np.arange(most_rated_movies_users_selection.shape[0]) , minor=False)
        ax.set_xticks(np.arange(most_rated_movies_users_selection.shape[1]) , minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        labels = most_rated_movies_users_selection.columns.str[:40]
        ax.set_xticklabels(labels, minor=False)
        ax.set_yticklabels(most_rated_movies_users_selection.index, minor=False)
        plt.setp(ax.get_xticklabels(), rotation=90)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    ax.grid(False)
    ax.set_ylabel('User id')
# Separate heatmap from color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
# Color bar
    cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
    cbar.ax.set_yticklabels(['5 stars', '4 stars','3 stars','2 stars','1 stars','0 stars'])
plt.show()
# Print the heatmap
draw_movies_heatmap(most_rated_movies_users_selection)

'''To understand this heatmap:
Each column is a different movie.
Each row is a different user.
The cell’s color is the rating that each user has given to each film. The values for each color can be checked in the scale of the right.
The white values correspond to users that haven’t rated the movie.
In order to improve the performance of the model, we’ll only use ratings for 1000 movies.'''

# Define Function to get the most rated movies
def get_users_who_rate_the_most(most_rated_movies, max_number_of_movies):
    # Get most voting users
    # 1- Count
    most_rated_movies['counts'] = pd.Series(most_rated_movies.count(axis=1))
    # 2- Sort
    most_rated_movies_users = most_rated_movies.sort_values('counts', ascending=False)
    # 3- Slice
    most_rated_movies_users_selection = most_rated_movies_users.iloc[:max_number_of_movies, :]
    most_rated_movies_users_selection = most_rated_movies_users_selection.drop(['counts'], axis=1)
    
    return most_rated_movies_users_selection


# Pivot the dataset and choose the first 1000 movies
user_movie_ratings =  pd.pivot_table(ratings_title, index='userid', columns= 'title', values='ratings')
most_rated_movies_1k = get_most_rated_movies(user_movie_ratings, 1000)

# Conversion to sparse csr matrix
sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())

'''Prediction
Now we will choose a cluster analyze it and try to make a prediction with it.'''

# Pick a cluster ID from the clusters above
cluster_number = 11
# Let's filter to only see the region of the dataset with the most number of values 
n_users = 75
n_movies = 300
cluster = clustered[clustered.group == cluster_number].drop(['index', 'group'], axis=1)
# Sort and print the cluster
cluster = sort_by_rating_density(cluster, n_movies, n_users)
draw_movies_heatmap(cluster, axis_labels=False)

#And now we will show the ratings:
cluster.fillna('').head()

# Fill in the name of the column/movie. e.g. 'Forrest Gump (1994)'
movie_name = "Matrix, The (1999)"
cluster[movie_name].mean()

# The average rating of 20 movies as rated by the users in the cluster
cluster.mean().head(20)

'''When a user logs in to our app, we can now show them recommendations that are appropriate to their taste. 
The formula for these recommendations is to 
select the cluster’s highest-rated movies that the user did not rate yet.'''

# Pick a user ID from the dataset
user_id = 10
# Get all this user's ratings
user_2_ratings  = cluster.loc[user_id, :]
# Which movies did they not rate? 
user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnull()]
# What are the ratings of these movies the user did not rate?
avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]
# Let's sort by rating so the highest rated movies are presented first
avg_ratings.sort_values(ascending=False)[:20]









    


    


    





