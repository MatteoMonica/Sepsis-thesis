import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import roc_auc_score,average_precision_score, accuracy_score, f1_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#confronto y_true(reale) con le y_pred del modello per vedere quanto "bravo" è il modello  
def evaluetion_metrics(y_true,y_pred):
    #quanto bene il modello distingue settici da non 
    auroc=roc_auc_score(y_true,y_pred)
    #misura precisione e sensibilità
    auprc=average_precision_score(y_true,y_pred)
    #quante predizioni sono corrette(sul totale)
    accuracy=accuracy_score(y_true,y_pred)
    #bilancia i falsi positivi e i falsi negativi 
    f1=f1_score(y_true,y_pred)
    print("\nAUROC",auroc,"\n\nAUPRC",auprc,"\n\nAccuracy",accuracy,"\n\nF1 Score",f1,"\n")
    return{"AUROC":auroc,"AUPRC":auprc,"Accuracy":accuracy,"F1 Score":f1}

#sto calcolando i punteggi da dare come nel paper
def calcola_punteggio(hours_to_sepsis,prediction):
    if pd.isna(hours_to_sepsis):
        if prediction == 1:
            return -0.05
        else:
            return 0
    else:
        if hours_to_sepsis> 12:
            if prediction == 1:
                return -0.05
            else:
                return 0
        elif hours_to_sepsis >= 6 and hours_to_sepsis <= 12: 
            if prediction == 1:
                return (12-hours_to_sepsis)/6
            else:
                return 0      
        elif hours_to_sepsis>=0 and hours_to_sepsis <=6:
            if prediction == 1:
                return  hours_to_sepsis / 6
            else:
                return 0
        elif hours_to_sepsis < 0:
            if prediction == 1:
                return -2
            else:
                return -2    


#leggo il mio file 
file= pd.read_csv("sepsis3_hourly_labeled.csv")

#converto sofa_time, suspected_infection_time e intime in formato datetime per poterli usare nei calcoli 
file["sofa_time"] = pd.to_datetime(file["sofa_time"], errors="coerce")
file["suspected_infection_time"] = pd.to_datetime(file["suspected_infection_time"], errors="coerce")
file["intime"] = pd.to_datetime(file["intime"], errors="coerce")

#l'onset della sepsi è il minimo tra sofa_time e suspected_infection_time, come definito nel paper
file["sepsis_onset"] = file[["sofa_time","suspected_infection_time"]].min(axis=1)

#calcolo a quante ore dall'ammissione in ICU avviene la sepsi
file["sepsis_onset_hour"] = (file["sepsis_onset"] - file["intime"]).dt.total_seconds() / 3600

#calcolo quante ore mancano alla sepsi per ogni riga (negativo = sepsi già avvenuta)
file["hours_to_sepsis"] = file["sepsis_onset_hour"] - file["hour_index"]

#controllo i valori unici 
pazienti = file["subject_id"].unique()

#mescolo i dati 
np.random.shuffle(pazienti)

#prendo il numero totale dei pazienti e calcolo i tagli da fare 
x=int(len(pazienti)*0.60)
y=int(len(pazienti)*0.80)

#faccio lo split, al training assegno il 60%, al validation 20% e al test 20%
train_ids=pazienti[0:x]
val_ids=pazienti[x:y]
test_ids=pazienti[y:]

#prendo le righe dei pazienti che appartengono al corrispettivo gruppo 
test_set=file[file["subject_id"].isin(test_ids)]
validation_set=file[file["subject_id"].isin(val_ids)]
train_set=file[file["subject_id"].isin(train_ids)]

#tolgo le colonne identificative e la label dalla X, la Y contiene solo la label, servirà per confrontare le predizioni del modello con la realtà
X_train=train_set.drop(["subject_id","hadm_id","stay_id","label_sepsis_6h","Gender","hour_start","hour_end","intime","antibiotic_time","culture_time","suspected_infection_time","sofa_time","sepsis3","sepsis_onset","is_sepsis","sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2","TroponinI"],axis=1)
Y_train=train_set["label_sepsis_6h"]
X_val=validation_set.drop(["subject_id","hadm_id","stay_id","label_sepsis_6h","Gender","hour_start","hour_end","intime","antibiotic_time","culture_time","suspected_infection_time","sofa_time","sepsis3","sepsis_onset","is_sepsis","sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2","TroponinI"],axis=1)
Y_val=validation_set["label_sepsis_6h"]
X_test=test_set.drop(["subject_id","hadm_id","stay_id","label_sepsis_6h","Gender","hour_start","hour_end","intime","antibiotic_time","culture_time","suspected_infection_time","sofa_time","sepsis3","sepsis_onset","is_sepsis","sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2","TroponinI"],axis=1)
Y_test=test_set["label_sepsis_6h"]


#Preparo e alleno il modello XGBoost
model= XGBClassifier()
model.fit(X_train,Y_train)
t=model.predict(X_val)
t_test_xgb=model.predict(X_test)
evaluetion_metrics(Y_val,t)
evaluetion_metrics(Y_test,t_test_xgb)

#Preparo la MLP,ovvero preparo i vari layer
mlp=Sequential()
X_train.shape[1]
scaler = StandardScaler()
imputer = SimpleImputer(strategy="mean")
X_train_clean = X_train.replace(-1, np.nan)
X_train_imputed = imputer.fit_transform(X_train_clean)
X_train_scaled = scaler.fit_transform(X_train_imputed)#devo normalizzare i valori per evitare che dominino
X_val_clean = X_val.replace(-1, np.nan)
X_val_imputed = imputer.transform(X_val_clean)
X_val_scaled = scaler.transform(X_val_imputed)#devo normalizzare i valori per evitare che dominino
X_test_clean = X_test.replace(-1, np.nan)
X_test_imputed = imputer.transform(X_test_clean)
X_test_scaled = scaler.transform(X_test_imputed)


#aggiungo il primo layer
mlp.add(Dense(units=64, activation="relu", input_dim=X_train_scaled.shape[1]))

#questo è l'hidden layer che elabora i dati
mlp.add(Dense(units=32,activation="relu"))

#output layer
mlp.add(Dense(units=1,activation="sigmoid"))

#qua sto specificando al modello come imparare
mlp.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])#il primo elemento aggiusta i pesi della rete dopo ogni errore, poi c'è la funzione che misura quanto sbagliata è la predizione,l'accuracy mi dice quanto sta migliorando

#alleno il modello 
mlp.fit(X_train_scaled,Y_train,epochs=50,batch_size=256,)
t_mlp=(mlp.predict(X_val_scaled)>0.5).astype(int)#NB keras a differenza di XGBoost mi restituisce non 0/1 ma bensi dei valori, facendo cosi converto in true/false quando >50% e poi converto in 0/1
t_test_mlp=(mlp.predict(X_test_scaled)>0.5).astype(int)
evaluetion_metrics(Y_val,t_mlp)
#valutazione sul test set
evaluetion_metrics(Y_test,t_test_mlp)

#faccio assegnare i punteggi ai modelli (in ore c'è il valore di hours_to sepsis, in pred la prediction) con zip le metto assieme, t sono le predizioni di XGBoost 
punteggi_xgb=[calcola_punteggio(ore,pred) for ore,pred in zip(validation_set["hours_to_sepsis"],t)]
punteggi_mlp=[calcola_punteggio(ore,pred) for ore,pred in zip(validation_set["hours_to_sepsis"],t_mlp)]

punteggi_xgb_test=[calcola_punteggio(ore,pred) for ore,pred in zip(test_set["hours_to_sepsis"],t_test_xgb)]
punteggi_mlp_test=[calcola_punteggio(ore,pred) for ore,pred in zip(test_set["hours_to_sepsis"],t_test_mlp)]

#stampo le medie 
print("Media utilità clinica XGBoost:", sum(punteggi_xgb) / len(punteggi_xgb))
print("Media utilità clinica MLP:", sum(punteggi_mlp) / len(punteggi_mlp))

print("Media utilità clinica XGBoost Test set:", sum(punteggi_xgb_test) / len(punteggi_xgb_test))
print("Media utilità clinica MLP Test set:", sum(punteggi_mlp_test) / len(punteggi_mlp_test))