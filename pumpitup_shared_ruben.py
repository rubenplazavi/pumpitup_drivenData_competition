import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', None)#Con este comando hacemos que los pd.head muestren todas las columnas (para un número podemos sustituir el none por cualquier cifra)

# Importamos los datos
df_train = pd.read_csv("training_values_pumpitup.csv")
y_df_train = pd.read_csv("training_labels_pumpitup.csv.csv")
df_test = pd.read_csv("test_values_pumpitup.csv")
#
# Comprobamos los head de cada columna
print("train data: \n", df_train.head(5))
print("Y labels: \n", y_df_train.head(5))
print("test data: \n", df_test.head(5))
#
# Comprobamos la cantidad de lineas y columnas de cada dataset
print("train data: \n", df_train.shape)
print("Y labels: \n", y_df_train.shape)
print("test data: \n", df_test.shape)
#
# Convirtiendo la Y en numérico
le = LabelEncoder()
y = le.fit_transform(y_df_train['status_group'].values)
y_df_train['status_group'] = y      # 0 --> funcional        # 1 --> necesita reparar       # 2 --> no funcional
#

# # ------ Combinamos train y la variable objetivo ------
# # hacemos una copia de train
training_replica = df_train.copy()
# # combinamos la Y y train
training_replica= pd.merge(training_replica, y_df_train, how='left', on='id')
# # hacemos una copia de test
testing_replica = df_test.copy()
# # Generamos nueva variable en cada dataset para luego poder separarlo si es necesario
training_replica['archivo'] = 0
testing_replica['archivo'] = 1
# # Combinamos la union de train/labels con test para generar un único dataset
all_data = pd.concat([training_replica, testing_replica], axis=0, copy=True)

#
# # ------------------------- Procesado del dataset -----------------------------
# # Comprobamos valores únicos
print("Valores únicos: \n", all_data.nunique())
#
# # Comprobamos valores nulos
print("Nulos:\n", all_data.isnull().sum())

# #Podemos dropear la columna scheme_name ya que scheme_management tiene los mismos datos y menos nulls
all_data = all_data.drop(["scheme_name"],axis=1)

# #convertimos a datetime la variable fecha para extraer el mes,año y dia
date = pd.to_datetime(all_data['date_recorded'], format='%Y-%m-%d')
year_recorded = date.dt.year
mont_recorded = date.dt.month
day_recorded = date.dt.day
all_data["year"] = year_recorded
all_data["month"] = mont_recorded
all_data["day"] = day_recorded
all_data = all_data.drop(["date_recorded"],axis=1)
###

# # Pasamos todos los valores que no deberían existir a nulos
null_values = {'year':0,
               'month':0,
               'day':0,
               'longitude':0,
               'latitude': -2.000000e-08,
               'population':0,
               'construction_year':0}

for columna, nulos in null_values.items():
    all_data[columna] = all_data[columna].replace(nulos,np.nan)

##<--------Transformamos los nulos-------->

#Para la variable funder
all_data["funder"] = all_data["funder"].fillna("Unknow")

#Para installer
all_data["installer"] = all_data["installer"].fillna("Unknow")

#Para Longitud y latitud
all_data["latitude"] = all_data["latitude"].fillna(-6.98)
all_data["longitude"] = all_data["longitude"].fillna(33)

#Para subvillage
#Hay valores que se han registrado muy mal por lo que borro esta columna
all_data = all_data.drop(["subvillage"],axis=1)

#Para population
all_data["population"] = all_data["population"].fillna(281)

#Para construction year
all_data["construction_year"] = all_data["construction_year"].fillna(all_data["construction_year"].mean())

#Para public_meeting
all_data["public_meeting"] = all_data["public_meeting"].fillna(True)

#Para scheme_management
all_data["scheme_management"] = all_data["scheme_management"].fillna('Other')


#Para permit
all_data["permit"] = all_data["permit"].fillna(True)

#
# # Eliminamos aquellas columnas que no aportan a nuestro dataset
all_data = all_data.drop(columns=['quantity_group', 'source_type', 'source_class', 'payment_type', 'extraction_type_group', 'extraction_type_class',
                                   'waterpoint_type_group', 'recorded_by',
                                  'num_private', 'wpt_name'])


##Convertimos variables booleanas y categóricas
print(all_data.dtypes)
#categóricas
categoricos = all_data.select_dtypes(include=["object"])
for i in categoricos.columns:
    all_data[i] = all_data[i].astype('category')
    all_data[i] = all_data[i].cat.codes

#booleanos
all_data["permit"] = all_data["permit"].astype(int)
all_data["public_meeting"] = all_data["public_meeting"].astype(int)

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
    lat_media = all_data.iloc[i,33]
    long_media = all_data.iloc[i,32]
    pto_med = (lat_media,long_media)
    lat = all_data.iloc[i,6]
    long = all_data.iloc[i,5]
    pto = (lat,long)
    #print("------->pto_med", pto_med, "pto", pto)
    # all_data['distance'] = distance.distance(pto_med, pto).km
    distance_arr.append(distance.distance(pto_med, pto).km)
all_data['distance'] = distance_arr

# from datetime import date
# current_year = date.today().year
# all_data["pump_year"] = current_year -all_data["construction_year"]

# # ------------------------------------------------------------------------------------------------------
# # ******************************************* MODELAMOS *************************************************

# separamos el dataset nuevamente para tener ya los datos train y test
train = all_data[all_data['archivo']==0]
test = all_data[all_data['archivo']==1]

X = train  # Datos con el label (variable objetivo)
X_predict = test  # Datos para la competición (no tiene variable objetivo)

#Eliminamos las columnas que habiamos generado en los dataset para luego separarlos
X = X.drop(columns=['archivo',"status_group"])
X_predict = X_predict.drop(columns=['archivo',"status_group"])

# # Escalamos los datos ya que nos da mejor resultado
X_std = StandardScaler().fit_transform(X)
# # Separamos nuestros datos (X e Y) en X_train, X_test, y_train, y_test por medio del train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=143)

# # el modelo usado es un xgBoost con unos parámetros concretos
modelxgb = XGBClassifier(objective = 'multi:softmax', booster = 'gbtree',
                      num_class = 3, eval_metric = 'merror', eta = .1,
                      max_depth = 14, colsample_bytree = .4)
# # Entrenamos el modelo
modelxgb.fit(X_train, y_train)

# # Predecimos con el modelo ya entrenado
y_pred = modelxgb.predict(X_test)
#generamos la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy", accuracy_score(y_test, y_pred))

# Generamos los datos de label
y_submision = modelxgb.predict(X_predict)

# # Para obtener la conversión inversa
y_submision = le.inverse_transform(y_submision)

print(y_submision)

## IMPRIMIR TABLA DE CORRELACION DE DATOS Y SU IMPORTANCIA CON RESPECTO A LA VARIABLE OBJETIVO
print("*********** Tabla de Correlacion con Variable Objetivo y su importancia con respecto a Ã©sta *************")
print(all_data.columns)
corr = abs(train.corr())
print(corr[['status_group']].sort_values(by = 'status_group',ascending = False))

submision = pd.read_csv("SubmissionFormat.csv")
submision["status_group"] = y_submision
submision.to_csv("submision_Conjunto.csv", index=False)

corr = all_data.corr()
(corr
    .status_group
    .drop('status_group') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())
plt.show()

'''
****************************** CONCLUSIONES: *************************************
Se ha estudiado los null, se han imputado con el valor medio de la región, se han tratado los outliers que hay, 
que son valores con valor 0 o los que hay en 'latitude': -2.000000e-08 (creando una lista para automatizar el proceso), 
se ha hecho feature ingenieering creandose el año de cada bomba, para ello se sacó el año poniendo la variable de año de 
construccion, la distancia al punto medio del pais usando la longitud y latitud, se ha probado con otras variables nuevas
que se han terminado descartando. 
Como cunclisión se observa que las variables más importantes para nuestro modelo han estado relacionadas con la latitud,
(que aunque no lo muestra, si que aporta una mejora cuantitativa), el año de construcción, waterpoint_type, quality_group, 
quantity, gps_height, region_code, extraction_type, source..
el mejor resultado en la competicion ha sido de 8224, quedando en la posicion 801
'''