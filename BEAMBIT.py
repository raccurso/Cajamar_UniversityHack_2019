import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
import os


##############################################################################
# Funcion para eliminar los outliers del train dataset.
# El metodo utilizado es eliminar aquellos ejemplos cuya diferencia respecto 
# al promedio sea mayor de un factor (2 o 3) por la desviacion estandard.
# 
# Parametros en entrada:
#   df = Dataframe con los datos de entrenamiento
#   col = nombre de la columna a evaluar
#   factor = multiplicador de la desviacion estandar (2 o 3)
#
# Salida:
#   Dataframe sin los outliers de la columna "col"
#
##############################################################################
def dropOutliers(df, col, factor):
    mean = df[col].mean()
    std = df[col].std()
    # Elimino las filas cuyo valor sea > promedio + factor * desviacion estandar
    df.drop(df[df[col] > (mean + factor * std)].index, inplace=True)
    
    # Elimino las filas cuyo valor sea < promedio - factor * desviacion estandar
    df.drop(df[df[col] < (mean - factor * std)].index, inplace=True)
    
    # Restituyo el dataframe sin los outliers
    return df


##############################################################################
# Funcion para transformar el dataset en entrada en un dataset util al modelo.
# Realiza las siguientes operaciones:
#   1. Creacion de features adicionales (ejemplo: numero de fotos)
#   2. Sustitucion de las variables categoricas mediante One hot encoding
#   3. Sustitucion de los NA
#   4. Composicion del dataset final con las features obtenidas
#   5. Normalizacion de las features
# 
# Parametros en entrada:
#   df = Dataframe de entrenamiento, test o prediccion
#   cp_dict = diccionario con los codigos postales a sustituir
#
# Salida:
#   Dataframe listo para el modelo de prediccion
#
##############################################################################
def preparaDataset(df, cp_dict, valoresMedios):
    # 1. Creacion de features adicionales
    # Calculo la longitud del texto en las columnas descripcion y distribucion.
    # Esta informacion influye en el tiempo de visualizacion del inmueble.
    # Sustituyo los NA con 0 en vez del promedio, ya que es necesario distinguir
    # los inmuebles que no poseen descripcion.
    colDesc = df['HY_descripcion'].str.len()
    colDesc.fillna(0, inplace=True)
    colDist = df['HY_distribucion'].str.len()
    colDist.fillna(0, inplace=True)

    # Calculo la cantidad de fotos que posee cada inmueble.
    # Esta informacion tambien deberia influir en el tiempo di visualizacion de la pagina.
    # Las fotos deben esta en el subdirectorio "imagenes_inmuebles_haya"
    fotos = os.listdir('imagenes_inmuebles_haya')
    
    # Obtengo una lista con los nombres de todos los archivos en el directorio. 
    # Me quedo con los 7 digitos iniciales, o sea el ID del inmueble al cual pertenece
    # la foto.
    codFotos = [int(x[0:7]) for x in fotos] 
    
    # Creo un dataframe con la cantidad de veces que aparece el ID de cada inmueble
    # en la lista de fotos. 
    cantidadFotos = pd.DataFrame([codFotos.count(x) for x in df['HY_id']], columns = ['CantidadFotos'])

    # 2. Sustitucion de las variables categoricas mediante One hot encoding.
    # Para el codigo postal, sustituyo los valores menos frecuentes con la constante "otro".
    # De esta manera reduzco la cantidad de features del dataset.
    # El diccionario "cp_dict" contiene los valores a sustituir.
    colCodPostal = df['HY_cod_postal'].replace(cp_dict)
    colCodPostal = pd.get_dummies(colCodPostal, prefix='CP_', drop_first=True)
    
    # La antiguedad es un campo numerico discreto, por lo cual conviene tratarlo 
    # como una variable categorica. 
    antiguedad = df['HY_antiguedad'].fillna(0)
    antiguedad = pd.to_numeric(antiguedad, downcast='integer')
    
    # Para las demas features, solo las transformo con One Hot Encoding
    colTipo = pd.get_dummies(df['HY_tipo'], prefix='TI_', drop_first=True)
    colProv = pd.get_dummies(df['HY_provincia'], prefix='PR_', drop_first=True)
    colAntiguedad = pd.get_dummies(antiguedad, prefix='AN_', drop_first=True)    
    colBanos = pd.get_dummies(df['HY_num_banos'], prefix='BA_', drop_first=True)
    colTerrazas = pd.get_dummies(df['HY_num_terrazas'], prefix='TE_', drop_first=True)
    certif = df['HY_cert_energ'].fillna('')
    colCertif = pd.get_dummies(certif, prefix='CE_', drop_first=True)

    # 3. Sustitucion de los NA
    # El precio anterior es la unica feature numerica para la que sustituyo los NA con 0.
    # Es util diferenciar los ejemplos que no tienen un precio anterior porque la 
    # variacion de precio puede influi en la duracion de la visita.
    colPrecioAnt = df['HY_precio_anterior'].fillna(0)    
    
    # En las demas features numericas sustituyo los NA con el promedio para no
    # influir en la prediccion.
    colMetrosTot = df['HY_metros_totales'].fillna(valoresMedios['HY_metros_totales'])
    colMetrosUt = df['HY_metros_utiles'].fillna(valoresMedios['HY_metros_utiles'])
    colPoblacion = df['IDEA_poblacion'].fillna(valoresMedios['IDEA_poblacion'])
    colArea = df['IDEA_area'].fillna(valoresMedios['IDEA_area'])
    colDensidad = df['IDEA_densidad'].fillna(valoresMedios['IDEA_densidad'])
    colPcComercio = df['IDEA_pc_comercio'].fillna(valoresMedios['IDEA_pc_comercio'])
    colPcIndustria = df['IDEA_pc_industria'].fillna(valoresMedios['IDEA_pc_industria'])
    colPcOficina = df['IDEA_pc_oficina'].fillna(valoresMedios['IDEA_pc_oficina'])
    colPcOtros = df['IDEA_pc_otros'].fillna(valoresMedios['IDEA_pc_otros'])
    colPcResid = df['IDEA_pc_residencial'].fillna(valoresMedios['IDEA_pc_residencial'])
    colPcPark = df['IDEA_pc_trast_parking'].fillna(valoresMedios['IDEA_pc_trast_parking'])
    colIndTienda = df['IDEA_ind_tienda'].fillna(valoresMedios['IDEA_ind_tienda'])
    colIndTurismo = df['IDEA_ind_turismo'].fillna(valoresMedios['IDEA_ind_turismo'])
    colIndAlim = df['IDEA_ind_alimentacion'].fillna(valoresMedios['IDEA_ind_alimentacion'])
    colIndRiqueza = df['IDEA_ind_riqueza'].fillna(valoresMedios['IDEA_ind_riqueza'])
    colRentAlq = df['IDEA_rent_alquiler'].fillna(valoresMedios['IDEA_rent_alquiler'])
    colIndElast = df['IDEA_ind_elasticidad'].fillna(valoresMedios['IDEA_ind_elasticidad'])
    colIndLiq = df['IDEA_ind_liquidez'].fillna(valoresMedios['IDEA_ind_liquidez'])
    colUnitPriRes = df['IDEA_unitprice_sale_residential'].fillna(valoresMedios['IDEA_unitprice_sale_residential'])
    colPriSelRes = df['IDEA_price_sale_residential'].fillna(valoresMedios['IDEA_price_sale_residential'])    
    colStockRes = df['IDEA_stock_sale_residential'].fillna(valoresMedios['IDEA_stock_sale_residential'])
    colDemandaRes = df['IDEA_demand_sale_residential'].fillna(valoresMedios['IDEA_demand_sale_residential'])
    colPrecioRes = df['IDEA_unitprice_rent_residential'].fillna(valoresMedios['IDEA_unitprice_rent_residential'])
    colPrecioRent = df['IDEA_price_rent_residential'].fillna(valoresMedios['IDEA_price_rent_residential'])
    colDemandaRent = df['IDEA_demand_rent_residential'].fillna(valoresMedios['IDEA_demand_rent_residential'])
    colStockRent = df['IDEA_stock_rent_residential'].fillna(valoresMedios['IDEA_stock_rent_residential'])

    # 4. Composicion del dataset final con las features obtenidas
    # Creo un nuevo dataframe con todas las columnas transformadas en los pasos anteriores.
    X_Orig = pd.concat([colTipo, colCodPostal,  colProv, colDesc, colMetrosTot, colMetrosUt, colBanos, cantidadFotos,
                        colTerrazas, colCertif, colPrecioAnt, colPoblacion, colArea, colDensidad,
                        colPcComercio, colPcIndustria, colPcOficina, colPcOtros, colPcResid, colPcPark,
                        colIndTienda, colIndTurismo, colIndAlim, colIndRiqueza, colRentAlq, colIndElast,
                        colIndLiq, colUnitPriRes, colPriSelRes, colStockRes, colDemandaRes, colPrecioRes, 
                        colPrecioRent, colDemandaRent, colStockRent,
                        df[['HY_ascensor', 'HY_trastero','HY_num_garajes', 'HY_precio', 'GA_page_views',
                                'GA_mean_bounce', 'GA_exit_rate', 'GA_quincena_ini', 'GA_quincena_ult']]], axis=1)

    # 5. Normalizacion de las features.
    # El dataframe contiene features numericas con escalas diferentes, por lo cual
    # es necesario normalizarlo para obtener valores entre 0 y 1 en todos.    
    array_x = X_Orig.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_norm = min_max_scaler.fit_transform(array_x)
    X = pd.DataFrame(data=x_norm, columns=X_Orig.columns.values)

    # Restituyo el dataframe final
    return X

##############################################################################
# Funcion para adaptar el dataset de test o prediccion a la misma 
# estructura del dataset de entrenamiento.
# Esta operacion es necesaria porque las variables categoricas transformadas
# con el One Hot Encoding, pueden generar un numero diferente de columnas en
# los dataframe. Esto depende de cuales valores se encuentran en las filas.
# Para que el dataframe de test o prediccion tenga las misma columnas que el
# de training, se realizan las siguientes operaciones:
# 1. Agregar al dataframe las columnas que faltan
# 2. Eliminar del dataframe las columnas que no existen en el dataset de training
# 
# Parametros en entrada:
#   x_train = Dataframe de entrenamiento, test o prediccion
#   x_test = Dataframe de test o prediccion
#
# Salida:
#   Dataframe con las mismas columnas del datafrane de training
#
##############################################################################
def completaColumnas(x_train, x_test):
    # 1. Agregar al dataframe las columnas que faltan
    for column in x_train:
        if (column not in x_test.columns):
            x_test[column] = 0

    # 2. Eliminar del dataframe las columnas que no existen en el dataset de training
    for column in x_test:
        if (column not in x_train.columns):
            x_test.drop(column, axis=1, inplace=True)
    
    # Restituyo el dataframe con la misma estructura del training
    return x_test


##############################################################################
# Funcion principal del programa.
# 
# Parametros en entrada:
#   modo = modo de ejecucion. Valores posibles:
#         'train_test'     --> Entrena el modelo y lo prueba para mostrar la MAE
#         'cross_valid'    --> Entrena y valida el modelo usando cross-validation
#         'train_predict'  --> Entrena el modelo y genera el archivo con las predicciones
#         'tune'           --> Utiliza el RandomizedSearchCV para buscar los hiperparametros
#                              que mejoran la prediccion.
#
##############################################################################
def main(modo):    
    # Cargo el CSV con los ejemplos para entrenar el modelo
    data = pd.read_csv("Modelar_UH2019.txt", sep='|')
    
    if modo == 'train_test':
        # En este modo, divido el dataset en 80% para entrenar y 20% para evaluar.
        # La subdivision del dataset la realizo desde un principio para no utilizar
        # ninguna informacion del la parte de test durante el entrenamiento del
        # modelo.
        df = data.sample(frac=1).reset_index(drop=True)
        df_train, df_test = train_test_split(df, test_size=0.2)
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
    else:
        # En los demas modos, utilizo todo el dataset para entrenar el modelo
        df_train = data

    # Elimino los outliers de las variables que mostraron valores muy distantes
    # del promedio o valores evidentemente equivocados (Ej. baños=99 o precio=0)
    dropOutliers(df_train, 'TARGET', 2)
    dropOutliers(df_train, 'HY_metros_totales', 2)
    dropOutliers(df_train, 'HY_num_banos', 2)
    dropOutliers(df_train, 'HY_precio', 3)
    df_train.drop(df_train[df_train['HY_precio'] == 0].index, inplace=True)

    # Elimino las filas sin provincia
    df_train.dropna(subset=['HY_provincia'], inplace=True)
    
    # Vuelvo a generar el indice del dataframe para evitar errores mas
    # adelante.
    df_train.reset_index(drop=True, inplace=True)
    
    # Memorizo todos los valores medios del dataset de train para utilizarlos
    # tambien durante el test
    valoresMedios = df_train.mean()

    # Para reducir el numero de features usando el one hot encoding sobre el codigo postal,
    # sustituyo los valores menos frecuentes con la constante "00000". El limite utilizado es 
    # de 0,3%, que en este dataset es aproximadamente 30 ejemplos.
    # Hipotizo  que los codigos postales menos frecuentes no dan beneficios al modelo.
    # El diccionario creado, cp_dict, sera utilizado en la funcion preparaDataset para
    # sustituir los valores.
    cp_count = df_train['HY_cod_postal'].value_counts()
    cp_dict = dict((c, '00000') for c in cp_count[cp_count < len(df_train)*0.003].index)

    # Preparo el dataset para entrenar el modelo X_train
    X_train = preparaDataset(df_train, cp_dict, valoresMedios)
    
    # Ordeno las columnas en orden alfabetico para que los 3 dataset (trainig, test y prediccion)
    # tengan siempre la misma estructura. De lo contrario, el algoritmo de entrenamiento
    # no funciona correctamente.
    X_train = X_train.reindex(sorted(X_train.columns), axis=1)
    
    # Memorizo en y_train la columna TARGET para entrenar el modelo
    y_train = df_train['TARGET']

    if modo == 'train_test':
        # En este modo, preparo el dataset de test para que tenga la misma
        # estructura del dataset de entrenamiento.
        X_test = preparaDataset(df_test, cp_dict, valoresMedios)
        X_test = completaColumnas(X_train, X_test)
        
        # Ordeno las columnas en orden alfabetico como en el dataset de entrenamiento
        X_test = X_test.reindex(sorted(X_test.columns), axis=1)
        
        # Memorizo en y_test la columna TARGET para probar el modelo
        y_test = df_test['TARGET']

    # Hiperparametros para ejecutar el algoritmo XGBRegressor.
    # Estos son los valores que arrojaron mejor resultado ejecutando el programa en 
    # modo "tune".
    xgb_params = {'base_score': 0.5,
                 'booster': 'gbtree',
                 'colsample_bylevel': 1,
                 'colsample_bytree': 1,
                 'gamma': 0.7,
                 'importance_type': 'gain',
                 'learning_rate': 0.1,
                 'max_delta_step': 0,
                 'max_depth': 6,
                 'min_child_weight': 7,
                 'missing': None,
                 'n_estimators': 30,
                 'n_jobs': 1,
                 'objective': 'reg:linear',
                 'reg_alpha': 0,
                 'reg_lambda': 5,
                 'scale_pos_weight': 1,
                 'seed': 0,
                 'silent': True,
                 'subsample': 1,
                 'alfa': 1,
                 'verbosity': 0}

    # Entreno el modelo utilizando el regresor de XGBoost
    model = xgb.XGBRegressor(params=xgb_params)    
    model.fit(X_train, y_train)
    
    if modo == 'train_test':
        # En este modo de ejecucion, realizo la prediccion sobre el dataset
        # de test para evaluar el modelo obtenido.
        y_pred = model.predict(X_test)

        # Calculo mediana de los errores absolutos para medir el modelo obtenido
        # sobre el dataset de prueba.
        MAE = median_absolute_error(y_test, y_pred)
        print('Mediana de los errores absolutos: ' + '{:.2f}'.format(MAE))

    if modo == 'tune':
        # Este modo de ejecucion tiene el objetivo de probar varios valores de 
        # hiperparametros y obtener el set de valores que obtenga mejor resultado.
        # Los valores presentes en params han sido modificados varias veces
        # para acercarme a la mejor parametrizacion.
        params = {'min_child_weight': [5, 7, 9],
                  'reg_lambda': [3, 5, 7],
                  'gamma': [0.5, 0.7],
                  'max_depth': [4, 6, 8],
                  'learning_rate':[0.1],
                  'n_estimators': [20, 25, 30]}

        # Creo el regresor de XGBoost
        model = xgb.XGBRegressor(objective='reg:linear', verbosity=0)
        
        # Utilizo GridSearchCV para probar todas las combinaciones de hiperparametros
        xgb_tune = GridSearchCV(estimator=model, param_grid =params, cv=5, verbose=2, n_jobs=2)
        xgb_tune.fit(X_train, y_train)
        
        # best_random contiene los hiperparametros con los que obtuve el mejor resultado
        best_random = xgb_tune.best_estimator_
        print(best_random)
    
    if modo == 'cross_valid':
        # En este modo evaluo el modelo a traves de la cross-validations en vez 
        # de dividir el dataset en training y test.
        # Para ello empleo el metodo k-fold
        MAE_promedio = 0
        # Divido el dataset en 5 partes, para hacer iteraciones con 80% / 20% entre training y test
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X_train):
            # Obtengo los dataset para esta iteracion
            X_train_k, X_test_k = X_train.values[train_index], X_train.values[test_index]
            y_train_k, y_test_k = y_train[train_index], y_train[test_index]        
            
            # Entreno el modelo de regresion
            model_k = xgb.XGBRegressor(params=xgb_params)                
            model_k.fit(X_train_k, y_train_k)
            
            # Pruebo el modelo de esta iteracion
            y_pred_k = model_k.predict(X_test_k)
            # Calculo mediana de los errores absolutos para medir el modelo obtenido
            # sobre el dataset de prueba.
            MAE_promedio = MAE_promedio + median_absolute_error(y_test_k, y_pred_k)
            
        # Al completar todas las iteraciones, calcolo el MAE promedio
        print('MAE Promedio: ' + '{:.2f}'.format(MAE_promedio/5)) 
    
    if modo == 'train_predict':
        # En este modo, entreno el modelo con todos los datos presentes en 
        # el archivo Modelar_UH2019.txt. Luego utilizo el modelo para predecir
        # sobre el dataset en el archivo Estimar_UH2019.txt.        
        dataEstimar = pd.read_csv('Estimar_UH2019.txt', sep='|')
        
        # Preparo el dataset estimar para que tenga la misma estructura del dataset
        # de entrenamiento.
        X_estim = preparaDataset(dataEstimar, cp_dict, valoresMedios)
        X_estim = completaColumnas(X_train, X_estim)
        
        # Ordeno las columnas en orden alfabetico como en el dataset de entrenamiento
        X_estim = X_estim.reindex(sorted(X_estim.columns), axis=1)

        # Ejecuto la prediccion con el dataset de estimar
        y_estim = model.predict(X_estim)        

        # Preparo el dataset final con los IDs de los inmuebles y el tiempo estimado
        dfTiempo = pd.DataFrame(data=y_estim, columns=['TM_Est'])
        dfFinal = pd.concat([dataEstimar['HY_id'], dfTiempo['TM_Est']], axis=1)    
        
        # Redondeo del tiempo a 2 decimales
        dfFinal.TM_Est = dfFinal.TM_Est.round(2)

        # Guardo el csv con las predicciones en el archivo BEAMBIT_UH2019.txt
        dfFinal.to_csv('BEAMBIT.txt', sep='|', encoding='utf-8', index=False)        


if __name__ == '__main__':
    # Modalidad de ejecucion:
    # modo = 'train_test'     --> Entrena el modelo y lo prueba para mostrar la MAE
    # modo = 'cross_valid'    --> Entrena y valida el modelo usando la cross-validation
    # modo = 'train_predict'  --> Entrena el modelo y genera el archivo con las predicciones
    # modo = 'tune'           --> Utiliza el GridSearchCV para buscar los hiperparametros
    #                             que mejoran la prediccion.   
    
    # Asigno a la variable "modo" la modalidad de ejecucion segun la leyenda arriba
    modo = 'train_predict'
    
    # Ejecuto el programa en el modo deseado
    main(modo)