import numpy as np
import pandas as pd
import seaborn as sns
import gc
import datetime
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_samples
from sklearn.model_selection import train_test_split
from yellowbrick.cluster import silhouette_visualizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import model_selection


# Import data
df_train = pd.read_csv("training_values_pumpitup.csv")
y_df_train = pd.read_csv("training_labels_pumpitup.csv.csv")
df_test = pd.read_csv("test_values_pumpitup.csv")

# # print table to chech if it has been read correctly
# print("train data: \n", df_train.head(5))
# print("Y labels: \n", y_df_train.head(5))
# print("test data: \n", df_test.head(5))
#
# # print the shape
# print("train data: \n", df_train.shape)
# print("Y labels: \n", y_df_train.shape)
# print("test data: \n", df_test.shape)

# Convert to label Y to a numeric variable
class_le = LabelEncoder()
y = class_le.fit_transform(y_df_train['status_group'].values)
y_df_train['status_group'] = y      # 0 --> functional        # 1 --> functional needs repair       # 2 --> not functional
# # Para obtener la conversión inversa
# class_le.inverse_transform(y)

# ------ COMBINE TRAIN, TEST AND LABELS TO DO COMMON FEATURE ENGINEERING ------
# make a copy of training data
training_replica = df_train.copy()
# we combine the labels with the
training_replica= pd.merge(training_replica, y_df_train, how='left', on='id')
training_replica.to_csv("training_replica.csv", index=False)
# A copy of the testing data
testing_replica = df_test.copy()

# set up a flag field to distinguish records from training and testing sets in the combined dataset
training_replica['tst'] = 0
testing_replica['tst'] = 1

# combine training and testing data into a single dataframe to do uniform part of feature engineering
all_data = pd.concat([training_replica, testing_replica], axis=0, copy=True)
all_data.to_csv("all_data.csv", index=False)
del training_replica
del testing_replica
gc.collect()

# ------------------------- Preprocesing both dataset together -----------------------------

print("unique values for all dataset: \n", all_data.nunique())

# ---------- HANDLING NULL VALUES AND OUTLIERS ----------

print("-------> Nulls:\n", all_data.isnull().sum())
# funder (4504) - installer (4532) - subvillage (470) - public_meeting (4155) - scheme_management (4846) - permit (3793)

# convert to datatime 'year_recorded' and create new variables year and month, drop date
date = pd.to_datetime(all_data['date_recorded'], format='%Y-%m-%d')
year_recorded = date.dt.year
mont_recorded = date.dt.month
print(year_recorded)

print("nulos para año construccion", all_data.loc[all_data['construction_year']==0])

all_data['year_recorded'] = year_recorded
all_data['month_recorded'] = mont_recorded
all_data = all_data.drop(columns='date_recorded')

all_data["years_old"] = 2022 - all_data["construction_year"]

# Create an array of numerical data to get the nulls (except 'region_code' and 'district_code', that has no 0 values)
numericals = [#'amount_tsh',
              'year_recorded',
              'month_recorded',
              #'gps_height',   # here the 0's could be at sea level, wich makes sense, so we keep them
              'longitude',
              'latitude',
              #'num_private',
              'population',
              'construction_year']

# looking for outliers on 'latitude' and 'longitude'
print(all_data.sort_values(by=['longitude'])['longitude'].head(10))
print(all_data.sort_values(by=['longitude'])['longitude'].tail(10))
print(all_data.sort_values(by=['latitude'])['latitude'].head(10))
print(all_data.sort_values(by=['latitude'])['latitude'].tail(10))

# dictionary with null values, in this case are all 0, we take them as nulls
null_values = {#'amount_tsh': 0,
               'year_recorded':0,
               'month_recorded':0,
               #'gps_height': 0,   # here the 0's could be at sea level, wich makes sense so we keep them
               'longitude': 0,  # 33
               'latitude': -2.000000e-08, #-6.98
               'population': 0,
               'construction_year': 0}
print("previous nulls: \n", all_data.isnull().sum())
# convert nulls 0, to nan's
all_data['construction_year']= all_data['construction_year'].astype(int)
for feature, null in null_values.items():
    all_data[feature] = all_data[feature].replace(null, np.nan)
# Imputing nans for numerical variables
# Replaces the NANs in a region with the mean of the other rows in that same region (which are much larger than wards)
for feature in numericals:
    replacements = all_data.groupby('district_code')[feature].transform('mean')
    all_data[feature] = all_data[feature].fillna(replacements)

cons = all_data.groupby(['management_group'])['construction_year'].transform("mean")
print("cons$$$$$$$$444444", cons)
all_data['construction_year'] = all_data['construction_year'].fillna(cons)
print("replacement: ------------------------\n", replacements )
print("sumatorio nulos: \n", all_data.isnull().sum())
# amount = all_data['amount_tsh']
# amount = np.log(amount+1)
# all_data['amount_tsh'] = amount
# amount = log(amount +1)

# all_data['dist_ecuador'] = np.sqrt(np.square(all_data["longitude"])+np.square(all_data["latitude"]))
# all_data=all_data.drop(columns=['longitude', 'latitude'])

#all_data['long_media'] = all_data.groupby('region')['longitude'].transform("mean")
#all_data['lat_media'] = all_data.groupby('region')['latitude'].transform("mean")

all_data['long_media'] = all_data['longitude'].mean()
all_data['lat_media'] = all_data['latitude'].mean()
# Imputing nans for categorical varibles (creating a new category "unknow")
all_data.fillna("unknow")

from geopy import distance
# newport_ri = (41.49008, -71.312796)
# cleveland_oh = (41.499498, -81.695391)
# print(distance.distance(all_data['lat_media', 'long_media'], all_data['latitude', 'longitude']).miles)

distance_arr = []
print("&&&&&&&&&&&&&&&&&&&&&", all_data.columns)
for i in range(0, len(all_data)):
    lat_media = all_data.iloc[i,44]
    long_media = all_data.iloc[i,43]
    pto_med = (lat_media,long_media)
    lat = all_data.iloc[i,6]
    long = all_data.iloc[i,5]
    pto = (lat,long)
    #print("------->pto_med", pto_med, "pto", pto)
    # all_data['distance'] = distance.distance(pto_med, pto).km
    distance_arr.append(distance.distance(pto_med, pto).km)
all_data['distance'] = distance_arr
print(all_data['distance'].head(100))
print(all_data['distance'].nunique())
#all_data = all_data.drop(columns = ['longitude','long_media', 'lat_media'])
# ------- TRANSFORM CATEGORICAL VARIABLE TO NUMERIC --------


print(" funder: \n")
print("unique values for 'waterpoint_type_group': \n", all_data['funder'].unique())
print("value counts for 'waterpoint_type_group': \n", all_data['funder'].value_counts())

# funder-> name of funder and number of repetitions
funder=all_data['funder'].value_counts()
# get the index(names) of the funders
name_funders=funder.loc[funder<10].index

all_data["funder"]= all_data["funder"].replace(name_funders, "small_funders")
#print("Small_funder\n", all_data.loc[all_data.funder=='small_funders'])

all_data['years'] = 2022 - all_data['construction_year']
#print("years: \n", all_data['years'])

# ------------ ENCODING / LABEL-ENCODING --------------
# Create a list with all the categorical data
categorical_data =["funder", "installer", "basin", "subvillage", "region", "lga", "ward", "public_meeting", "permit",
                   "scheme_management", "extraction_type", "management", "management_group",
                   "payment", "water_quality", "quantity", "source", "waterpoint_type"]

# bucle to encode all the categorical data with LabelEncoder
for categ in categorical_data:
    all_data[categ] = LabelEncoder().fit_transform(all_data[categ].values)

# 'quantity', 'quantity_group' -- 'source', 'source_type', 'source_class' -- 'payment', 'payment_type' -- 'extraction_type', 'extraction_type_group', 'extraction_type_class'
# 'waterpoint_type_group', 'waterpoint_type' -- 'water_quality', 'quality_group'
# 'scheme_name' (too many nulls, not many info) --- 'num_private' (almost all are 0.) -- 'recorded_by' (only 1 value)
# 'wpt_name'   (many diferent values, probably not than much info, heavy to digest..)
# we don't see much correlation between 'public_meeting'
all_data = all_data.drop(columns=['id', 'quantity_group', 'source_type', 'source_class', 'payment_type',
                                  'extraction_type_group', 'extraction_type_class', 'scheme_name',
                                  'waterpoint_type_group', 'recorded_by', 'quality_group',
                                  'num_private', 'wpt_name', 'status_group']) #,'public_meeting'
all_data.to_csv("prueba_long.csv", index=False)
print("all data:\n", all_data.columns)
# 'id', 'amount_tsh', 'funder', 'gps_height', 'installer', 'longitude', 'latitude', 'basin', 'subvillage', 'region', 'region_code',
# 'district_code', 'lga', 'ward', 'population', 'public_meeting', 'scheme_management', 'permit', 'construction_year', 'extraction_type',
# 'management', 'management_group', 'payment', 'water_quality', 'quality_group', 'quantity', 'source', 'waterpoint_type',
# 'status_group', 'tst', 'encoded_funder', 'encoded_basin', 'year_recorded', 'month_recorded'

# ------------------------------------------------------------------------------------------------------
# ******************************************* MODELING *************************************************

# split all_data DataFrame into training and testing again
X = all_data[all_data['tst']==0]  # data with label, for training and testing
X_predict = all_data[all_data['tst']==1]  # data to send to the competition
X = X.drop(columns=['tst'])
X_predict = X_predict.drop(columns=['tst'])
# we already encode the labels and create the y variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)#, stratify=y)

# #clf1 = LogisticRegression(random_state=1, max_iter=10000)
# clf2 = RandomForestClassifier(random_state=1)
# clf3 = GaussianNB()
# #eclf = EnsembleVoteClassifier(clfs=[clf1, clf2], weights=[1, 1])
#
# print('5-fold cross validation:\n')
# labels = ['Random Forest', 'Ensemble']
# for clf, label in zip([clf2, clf3], labels):
#     scores = model_selection.cross_val_score(clf, X, y,
#                                              cv=5,
#                                              scoring='accuracy')
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]"
#           % (scores.mean(), scores.std(), label))



from xgboost import XGBClassifier
modelxgb = XGBClassifier(objective = 'multi:softmax', booster = 'gbtree',
                      num_class = 3, eval_metric = 'merror', eta = .1,
                      max_depth = 14, colsample_bytree = .4)

print(X_train.shape)
print(y_train)
# modelxgb.fit(X_train, y_train)
#
# from sklearn.metrics import accuracy_score
# y_pred = modelxgb.predict(X_test)
# print("Accuracy", accuracy_score(y_test, y_pred))
#
# y_submision = modelxgb.predict(X_predict)
# # # Para obtener la conversión inversa
# y_submision = class_le.inverse_transform(y_submision)

# WITH STANDARISE DATA !!!!!!!!!!!!!!!
sc = StandardScaler()
sc.fit(X_train)   # se pasan los datos para que haga la desviacion tipica y la media de la varianza, para que estandarice estos datos..
X_train_std = sc.transform(X_train)  #se entrenan solo los de entrenamiento, pero transforman todos (train y test)
X_test_std = sc.transform(X_test)   # se transforman tambien los de test
X_predict_std = sc.transform(X_predict)

modelxgb.fit(X_train_std, y_train)

from sklearn.metrics import accuracy_score
y_pred = modelxgb.predict(X_test_std)
print("Accuracy", accuracy_score(y_test, y_pred))

y_submision = modelxgb.predict(X_predict_std)
# # Para obtener la conversión inversa
y_submision = class_le.inverse_transform(y_submision)

print(y_submision)
submision = pd.read_csv("SubmissionFormat.csv")
submision["status_group"] = y_submision
submision.to_csv("submision_ruben.csv", index=False)


'''
We observe that the data is not order by id, so we are gonna order by id, and use this column as an index
'''

del all_data
gc.collect()
