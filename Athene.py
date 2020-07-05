# import the required libraries.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kmodes.kmodes import KModes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import nanmedian
import random
from numpy import dot
from numpy.linalg import norm
import math

df = pd.read_csv("responses.csv")  # reading the data

# fill the nan values using the medians
df = df.groupby(df.columns, axis=1).transform(lambda x: x.fillna(x.median()))
# used to find the nan count of attributes


def nancount(x):
    count = 0
    a = df[x].isnull()  # returns true for null values
    for val in a:
        if(val):
            count += 1
    return count

# dictinary to store the nan counts with keys as attribute names
counts = {}
for i in df.columns:
    returnval = nancount(i)
    if(returnval != 0):
        counts[i] = nancount(i)
print(counts)


def replace(col):
    # find the unique values
    un = pd.unique(df[col])
    j = 1
    # dictionary used to map the categorical variables
    dict_1 = {}
    for i in un[:-1]:
        dict_1[i] = j
        j += 1
    # un[-1] has nan in it. Hence that must have a value of zero initially
    dict_1[un[-1]] = 0
    # print(dict_1)
    x = []
    # list containing the integer values of the categorical variales
    for index, row in df[col].iteritems():
        x.append((dict_1[row]))
    med = np.median(x)
    # the zero valued rows are replaced with the median
    for i in range(0, len(x)):
        if(x[i] == 0):
            x[i] = int(med)
    # assign it to the dataframe
    df[col] = x


NaN_Col = ['Smoking', 'Alcohol', 'Punctuality', 'Lying', 'Education']
for i in NaN_Col:
    replace(i)

#printing the nan counts
def nancount(x):
    count = 0
    a = df[x].isnull()
    for val in a:
        if(val):
            count += 1
    return count


counts = {}
for i in df.columns:
    returnval = nancount(i)
    if(returnval != 0):
        counts[i] = nancount(i)
print(counts)

# total no.of attributes
valcount = df.Gender.value_counts()
print(valcount)
# find the percentage of the population who are male
noperc = (valcount['female'] / (valcount['male'] + valcount['female'])) * 100
# find the percentage of the population who are female
yesperc = (valcount['male'] / (valcount['male'] + valcount['female'])) * 100
print(noperc, yesperc)
# pie chart to represent the ratio
plt.pie(valcount, labels=['female', 'male'], autopct='%1.0f%%')
plt.show()

# dealing with missing values present in the Gender column
for index, row in df.Gender.iteritems():
    if(isinstance(row, float)):
        # generate a random number
        x = random.randint(1, 100)
        # based on the ratio classify the missing values
        if (x >= 1 and x <= 60):
            df.Gender.loc[index] = 'female'
        else:
            df.Gender.loc[index] = 'male'

# similar way to handle the missing value for the column 'Left-right handed'
valcount = df['Left - right handed'].value_counts()
noperc = (valcount['right handed'] /
          (valcount['right handed'] + valcount['left handed'])) * 100
yesperc = (valcount['left handed'] /
           (valcount['right handed'] + valcount['left handed'])) * 100
print(noperc, yesperc)

plt.pie(valcount, labels=['right handed', 'left handed'], autopct='%1.0f%%')
plt.show()

for index, row in df['Left - right handed'].iteritems():
    if(isinstance(row, float)):
        x = random.randint(1, 100)
        if (x >= 1 and x <= 90):
            df['Left - right handed'].loc[index] = 'right handed'
        else:
            df['Left - right handed'].loc[index] = 'left handed'

# similar way to handle the missing value for the column 'Only Child'
valcount = df['Only child'].value_counts()
noperc = (valcount['no'] / (valcount['yes'] + valcount['no'])) * 100
yesperc = (valcount['yes'] / (valcount['yes'] + valcount['no'])) * 100
print(noperc, yesperc)

plt.pie(valcount, labels=['no', 'yes'], autopct='%1.0f%%')
plt.show()


for index, row in df['Only child'].iteritems():
    if(isinstance(row, float)):
        x = random.randint(1, 100)
        if (x >= 1 and x <= 75):
            df['Only child'].loc[index] = 'no'
        else:
            df['Only child'].loc[index] = 'yes'

# similar way to handle the missing value for the column 'Village-Town'
valcount = df['Village - town'].value_counts()
noperc = (valcount['village'] / (valcount['village'] + valcount['city'])) * 100
yesperc = (valcount['city'] / (valcount['village'] + valcount['city'])) * 100
print(noperc, yesperc)
plt.pie(valcount, labels=['City', 'Village'], autopct='%1.0f%%')
plt.show()

for index, row in df['Village - town'].iteritems():
    if(isinstance(row, float)):
        x = random.randint(1, 100)
        if (x >= 1 and x <= 70):
            df['Village - town'].loc[index] = 'city'
        else:
            df['Village - town'].loc[index] = 'village'

# similar way to handle the missing value for the column 'House - block of -flats'
valcount = df['House - block of flats'].value_counts()
noperc = (valcount['house/bungalow'] /
          (valcount['house/bungalow'] + valcount['block of flats'])) * 100
yesperc = (valcount['block of flats'] /
           (valcount['house/bungalow'] + valcount['block of flats'])) * 100
print(noperc, yesperc)

plt.pie(valcount, labels=['Flat', 'House'], autopct='%1.0f%%')
plt.show()

for index, row in df['House - block of flats'].iteritems():
    if(isinstance(row, float)):
        x = random.randint(1, 100)
        if (x >= 1 and x <= 60):
            df['House - block of flats'].loc[index] = 'block of flats'
        else:
            df['House - block of flats'].loc[index] = 'house/bungalow'

# Printing the nan counts


def nancount(x):
    count = 0
    a = df[x].isnull()
    for val in a:
        if(val):
            count += 1
    return count


counts = {}
for i in df.columns:
    returnval = nancount(i)
    if(returnval != 0):
        counts[i] = nancount(i)
print(counts)

# replacing 1 and 2 for binary variables


def replace1(col):
    un = pd.unique(df[col])# get the unique values
    j = 1
    #dictionary to assign numbers to each unique value
    dict_1 = {} 
    for i in un:
        dict_1[i] = j
        j += 1
    # print(dict_1)
    x = []
    for index, row in df[col].iteritems():
        x.append((dict_1[row]))
    df[col] = x


NaN_Col = [
    'Gender',
    'Left - right handed',
    'Only child',
    'Village - town',
    'House - block of flats',
    'Internet usage']
for i in NaN_Col:
    replace1(i)

med1 = np.nanmedian(df.Achievements)
for index, row in df.Achievements.iteritems():
    a = df.Achievements.isnull()
    for val in a:
        if(val):
            df.Achievements.loc[index] = med1


def nancount(x):
    count = 0
    a = df[x].isnull()
    for val in a:
        if(val):
            count += 1
    return count


counts = {}
for i in df.columns:
    returnval = nancount(i)
    if(returnval != 0):
        counts[i] = nancount(i)
print(counts)
###################################Visualization#########################

# histogram of age
sns.set(color_codes=True)
sns.distplot(df.Age)
plt.show()

# Histogram of age grouped by place of stay
fig, ax = plt.subplots(figsize=(10, 5))

var_of_int_ser = df['Village - town']
sns.distplot(df[var_of_int_ser == 1].Age.dropna(),
             label='village', ax=ax, kde=False, bins=30)

sns.distplot(df[var_of_int_ser == 2].Age.dropna(),
             label='city', ax=ax, kde=False, bins=30)
ax.legend()

# histogram of age grouped by gender
fig, ax = plt.subplots(figsize=(10, 5))

var_of_int_ser = df['Gender']
sns.distplot(df[var_of_int_ser == 1].Age.dropna(),
             label='female', ax=ax, kde=False, bins=30)

sns.distplot(df[var_of_int_ser == 2].Age.dropna(),
             label='male', ax=ax, kde=False, bins=30)
ax.legend()

# selecting the important variables for hobbies and interests in data and
# plotting a correlation grid
data = [
    'Reading',
    'Science and technology',
    'Dancing',
    'Writing',
    'Musical instruments',
    'Theatre',
    'Pets',
    'Fun with friends',
    'Adrenaline sports',
    'Darkness',
    'Snakes',
    'Horror',
    'Comedy',
    'Romantic',
    'Thriller',
    'Sci-fi',
    'Documentary',
    'Action',
    'War']
corr = df[data].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(100, 20, as_cmap=True)
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=.3,
    center=0,
    square=True,
    linewidths=.5)

# boxplot of Age (since the outliers aren't way out of bound, cleaning it
# is not needed)
plt.boxplot(df['Age'])
plt.ylabel('Age')
plt.show()

# boxplot of weight before cleaning
plt.boxplot(df.Weight)
plt.show()

# converting pound to kg as outliers is in pound (kg does not make sense)
for index, row in df.Weight.iteritems():
    if(row >= 100):
        df.Weight.loc[index] = row * 0.454

# boxplot of height before cleaning
plt.boxplot(df.Height)
plt.show()

# converting inch to cm as 60cm is not possible for a human
for index, row in df.Height.iteritems():
    if(row <= 130):
        df.Height.loc[index] = row * 2.54

# plot of height after replacing outliers
plt.boxplot(df.Height)
plt.show()

# plot of weight after replacing outliers
plt.boxplot(df.Weight)
plt.show()

# calculating bmi
df['BMI'] = round(df['Weight'] / ((df['Height'] / 100)**2), 1)

# plot for BMI vs Age
plt.bar(df['Age'], df['BMI'])
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()

# histogram, median, mean for Music
plt.hist(df.Music)
print(np.median(df.Music))
print(np.mean(df.Music))
plt.xlabel('Music rated based on likes')
plt.show()

plt.hist(df.Movies)
print(np.median(df.Movies))
print(np.mean(df.Movies))
plt.xlabel('Movies rated based on likes')
plt.show()

# splitting into training and test data for PCA
y = df.Movies
y1 = df[df.columns[0:19]]
y2 = df[df.columns[20:150]]
X = pd.concat([y1, y2], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# feature scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA function on training
# and testing set of X component

pca = PCA(n_components=120)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

'''this is to check the variance chanes based on number of components. Since after 120, not much change is there, we choose
that as n_components in PCA'''
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')  # for each component
plt.title('Responses Dataset Explained Variance')
plt.show()

###################################K-Means##############################

# Kmeans clustering for clustering based on Age

df_test = pd.DataFrame(columns=['Age', 'Movies'])
df_test['Age'] = df.Age
df_test['Movies'] = df.Movies
mat = df_test.values
# divide the data into 2 clusters
kmeans = KMeans(n_clusters=2, random_state=10)
kmeans.fit(mat)
labels = kmeans.labels_
results = pd.DataFrame([df_test.index, labels]).T
print(results)
# plotting a scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(df_test['Age'], df_test['Movies'], c=results[1], s=50)
plt.colorbar(scatter)

###############################Linear Regression#########################

# multiple linear regression

# regression on genre 'Comedy'
X = pd.DataFrame(np.c_[df.iloc[:, 31:73]])
y = df['Comedy']

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9)

lin_reg_mod = LinearRegression()

# fit the model
lin_reg_mod.fit(X_train, y_train)

# test the model
pred = lin_reg_mod.predict(X_test)

test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

test_set_r2 = r2_score(y_test, pred)

# error rates
print("rmse", test_set_rmse)
print("r-squared", test_set_r2)


##################################K-modes################################

# kmodes clustering on Age
# divide the data into 2 clusters
data = pd.DataFrame({'Movies': df['Movies'], 'Age': df['Age']})

km = KModes(n_clusters=2, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(data)
fig = plt.figure()
# plotting the scatter plot
ax = fig.add_subplot(111)
scatter = ax.scatter(data['Age'], data['Movies'], c=clusters, s=50)
plt.colorbar(scatter)

################################Model####################################

# function to calculate the cosine similarity


def sim(a, b):
    cos_sim = np.dot(a, b) / (norm(a) * norm(b))
    return cos_sim

# agglomerative clustering based on centroid distance


def trainModelAndValidate(train, test):
    count = 0
    # select the required columns
    per = pd.DataFrame(np.c_[train.iloc[:, 31:73]])
    # kmodes clustering with initial cluster as 500
    km = KModes(n_clusters=500, max_iter=1000, init='Huang', n_init=2,
                n_jobs=-1)
    print("Cost of K clusters")
    m1 = km.fit(per)
    # print the cost of clustering
    print("500 clusters:", m1.cost_)

    # reduce the clusters gradually till the cost is minimized
    mdl1 = m1.cluster_centroids_
    km1 = KModes(
        n_clusters=250,
        max_iter=1000,
        init='Huang',
        n_init=2,
        n_jobs=-1)
    m2 = km1.fit(mdl1)
    # print(m2.cluster_centroids_)
    print("250 clusters:", m2.cost_)

    mdl2 = m2.cluster_centroids_
    km2 = KModes(
        n_clusters=125,
        max_iter=1000,
        init='Huang',
        n_init=2,
        n_jobs=-1)
    m3 = km2.fit(mdl2)
    # print(m3.cluster_centroids_)
    print("125 clusters:", m3.cost_)

    mdl3 = m3.cluster_centroids_
    km3 = KModes(
        n_clusters=62,
        max_iter=1000,
        init='Huang',
        n_init=2,
        n_jobs=-1)
    m4 = km3.fit(mdl3)
    # print(m4.cluster_centroids_)
    print("62 clusters:", m4.cost_)

    mdl4 = m4.cluster_centroids_
    km4 = KModes(
        n_clusters=31,
        max_iter=1000,
        init='Huang',
        n_init=2,
        n_jobs=-1)
    m5 = km4.fit(mdl4)
    # print(m5.cluster_centroids_)
    print("31 clusters:", m5.cost_)

    mdl5 = m5.cluster_centroids_
    km5 = KModes(
        n_clusters=15,
        max_iter=1000,
        init='Huang',
        n_init=2,
        n_jobs=-1)
    m6 = km5.fit(mdl5)
    # print(m6.cluster_centroids_)
    print("15 clusters:", m6.cost_)

    mdl6 = m6.cluster_centroids_
    km6 = KModes(
        n_clusters=10,
        max_iter=1000,
        init='Cao',
        n_init=2,
        n_jobs=-1)
    m7 = km6.fit(mdl6)
    # print(m7.cluster_centroids_)
    print("10 clusters:", m7.cost_)

    mdl7 = m7.cluster_centroids_
    km7 = KModes(
        n_clusters=8,
        max_iter=1000,
        init='Cao',
        n_init=2,
        n_jobs=-1)
    m8 = km7.fit(mdl7)
    mfin_clust = m8.cluster_centroids_
    print("8 clusters:", m8.cost_)
    print()

    # The min cost is obtained when number of clusters = 8
    mfin = km7.fit_predict(per)
    fin = pd.DataFrame(mfin)
    # print(mfin_clust)

    # select the required columns
    df1 = train.iloc[:, 20:73]
    # add a new column which has the final classification
    df1['clusters'] = mfin
    # In order to find the similarity between the users, we group the users
    # who belong to the same cluster
    df_fin = df1.groupby(['clusters'])

    fin_0 = df_fin.get_group(0)
    # print(np.std(fin_0['Horror']))

    fin_1 = df_fin.get_group(1)
    # print(np.std(fin_1['Horror']))

    fin_2 = df_fin.get_group(2)

    fin_3 = df_fin.get_group(3)

    fin_4 = df_fin.get_group(4)

    fin_5 = df_fin.get_group(5)

    fin_6 = df_fin.get_group(6)

    fin_7 = df_fin.get_group(7)

    # convert the centroids of a cluster into a list
    mfin_clust = list(mfin_clust)

    for i in range((test.shape[0])):
        row_hobby = list(df.iloc[i, 31:73])
        row_genre = list(df.iloc[i, 20:31])

        # Euclidian distance between y and the centroid of each cluster
        # The calculated distances are stored in a dictionary with the key =
        # cluster numbers
        distance = {}

        for i in range(0, 8):
            distance[i] = (math.sqrt(
                sum([(a - b) ** 2 for a, b in zip(mfin_clust[i], row_hobby)])))
        # minimum distance is calculated using the values of the dictionary
        min_clust = min(distance, key=distance.get)
        # the user is classified into the cluster
        df_clust = df_fin.get_group(min_clust)
        # similarity for y and df_clust
        # drop the columns containing movie genre as the similarity between the
        # users is calculated using the hobbies preferences
        df_clust2 = df_clust.drop(['Horror',
                                   'Romantic',
                                   'Comedy',
                                   'Thriller',
                                   'Sci-fi',
                                   'War',
                                   'Fantasy/Fairy tales',
                                   'Western',
                                   'Animated',
                                   'Documentary',
                                   'Action'],
                                  axis=1)
        # insert a new column called index as each user needs to have one
        # unique identity
        ind = list(range(0, len(df_clust2)))
        df_clust2.insert(0, 'Index', ind)

        # add the index column to the dataframe
        df_clust2['Index']
        # dictinary to store the user-user similarity
        xz_dict = {}
        for j in range(0, len(df_clust2)):
            xz = []
            # the list contains the column header and the preferences of the
            # jth row
            xz = list(df_clust2.iloc[j, :].items())
            # print(xz)
            xz1 = []
            # append only the preferences in a new list
            for i in range(1, 43):
                xz1.append(xz[i][1])
            # print(xz1)
            simi = sim(xz1, row_hobby)
            # store the user similarity in a dictionary
            xz_dict[j] = simi
        # find 5 users who are most similar to the new user
        top_5 = sorted(xz_dict, key=xz_dict.get, reverse=True)[:5]

        # dictionary used to store the rating for each genre based on user
        # similarities
        fin_rec = {}
        actual = {}
        # for each genre
        for k in range(1, 12):
            actual[k] = row_genre[k - 1]
            user_rating = []
            sum_sim = 0
            rec = 0
            # apped the ratings of the similar users into a list for a
            # particular genre
            for i in top_5:
                user_rating.append(df_clust.iloc[i, k:k + 1].item())
            # calculate the rating of the new user based on user-user
            # similarity
            for i, j in zip(user_rating, xz_dict):
                rec = rec + (i * xz_dict[j])
                sum_sim = sum_sim + xz_dict[j]
            # store the rating in the dictionary created
            fin_rec[k] = rec / sum_sim
        # select the top 3 genres based on rating
        top_3 = sorted(fin_rec, key=fin_rec.get, reverse=True)[:3]
        top_3_actual = sorted(actual, key=actual.get, reverse=True)[:3]
        # Thus recommend the genres to the user.
        for l in top_3:
            if l in top_3_actual:
                count += 1
    print("Accuracy", count / (3 * test.shape[0]))
    return count / (3 * test.shape[0])


kf = KFold(5, True, 1)
i = 1
list_accuracy = []
for train, test in kf.split(df):
    print("______________________________________________")
    print("Fold" + str(i))
    i += 1
    list_accuracy.append(trainModelAndValidate(df.iloc[train], df.iloc[test]))
print("______________________________________________")
print(
    "Accuracy of the model is",
    (sum(list_accuracy) / len(list_accuracy)) * 100,
    "%")
