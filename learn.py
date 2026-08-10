import pandas as pd
import numpy as np
import os
import warnings
import random
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import roc_auc_score,average_precision_score, accuracy_score, f1_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping

#Creo la funzione per crearmi il dataset da dare alla LSTM perchè serve diverso rispetto a quello usato dagli altri modelli
def crea_sequenze(dataset,feature_cols,n_ore):
    X_sequenze = []
    Y_label = [] 
    for patient_id,group in dataset.groupby("subject_id"):
        group = group.sort_values("hour_index")
        for t in group["hour_index"]:  
            finestra = group[group["hour_index"] <= t]
            # prendo le ultime n_ore righe della finestra di osservazione fino all'ora t, selezionando solo le feature cliniche 
            sequenza = finestra.tail(n_ore)[feature_cols]
            if len(sequenza) < n_ore:
                #aggiungo zeri da mettere prima della sequenza per far si che tutti abbiano almeno 12 ore
                padding = np.zeros((n_ore - len(sequenza),len(feature_cols)))
                #concateno a sequenza gli zeri che ho aggiunto
                sequenza = np.concatenate([padding,sequenza.values],axis=0)
            else:
                sequenza = sequenza.values
            #creo le sequenze temporali dei pazienti con anche le loro feature cliniche 
            X_sequenze.append(sequenza)
            #prendo la label_sepsis_6h della riga che corrisponde all'ora t, voglio sapere se a quell'ora il paziente stava sviluppando sepsi
            Y_label.append(group[group["hour_index"] == t]["label_sepsis_6h"].values[0])
    return np.array(X_sequenze),np.array(Y_label) 

#Confronto y_true(reale) con le y_pred del modello per vedere quanto "bravo" è il modello,y_prob è la probabilità,AUROC e AUPRC le richiedono perchè misurano quanto bene il modello ordina i pazienti dal più al meno rischioso 
def evaluetion_metrics(y_true,y_pred,y_prob):
    #quanto bene il modello distingue settici da non 
    auroc=roc_auc_score(y_true,y_prob)
    #misura precisione e sensibilità
    auprc=average_precision_score(y_true,y_prob)
    #quante predizioni sono corrette(sul totale)
    accuracy=accuracy_score(y_true,y_pred)
    #bilancia i falsi positivi e i falsi negativi 
    f1=f1_score(y_true,y_pred)
    print("\nAUROC",auroc,"\n\nAUPRC",auprc,"\n\nAccuracy",accuracy,"\n\nF1 Score",f1,"\n")
    return{"AUROC":auroc,"AUPRC":auprc,"Accuracy":accuracy,"F1 Score":f1}

#Sto calcolando i punteggi da dare come nel paper (PhysioNet)
#NB: la finestra di reward/penalità per pazienti settici si estende da 12h prima dell'onset
#fino a 3h DOPO l'onset (non si azzera esattamente a hours_to_sepsis=0)
def calcola_punteggio(hours_to_sepsis,prediction,is_sepsis):
    if is_sepsis == False:
        if prediction == 1:
            return -0.05
        else:
            return 0
    else:
        if pd.isna(hours_to_sepsis):
            return 0  # settico ma senza hours_to_sepsis valido
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
        elif hours_to_sepsis>=-3 and hours_to_sepsis <6: #finestra estesa fino a -3 (prima si fermava a 0): un TP appena dopo l'onset va ancora premiato, non punito vedi grafico paper
            if prediction == 1:
                return  (hours_to_sepsis+3) / 9
            else:
                return -2 *(6-hours_to_sepsis)/9
        elif hours_to_sepsis < -3:
            if prediction == 1:
                return 0 #oltre le 3h dopo l'onset un TP non è più premiato, ma non è nemmeno penalizzato come un errore
            else:
                return -2    

#Sto normalizzando i punteggicome nel paper
def normalizza_punteggio(hours_to_sepsis_list,is_sepsis_list,prediction):
   #è il punteggio reale del modello con le sue predizioni
   U_totale = sum([calcola_punteggio(ore, pred, sepsi) for ore, pred, sepsi in zip(hours_to_sepsis_list, prediction, is_sepsis_list)])

   #questo è il punteggio nel caso peggiore in cui non predice mai sepsi(cioè il punto di partenza)
   U_no_predictions=sum([calcola_punteggio(ore, 0, sepsi) for ore, sepsi in zip(hours_to_sepsis_list, is_sepsis_list)])

   #questo è il punteggio se il modello predice sempre sepsi (cioè il massimo che posso avere)
   U_optimal=sum([calcola_punteggio(ore,1,sepsi) for ore,sepsi in zip(hours_to_sepsis_list, is_sepsis_list)])

   #qua sto facendo la normalizzazione cioè quanto si avvicina il modello al punteggio ottimale(optimal), partendo dal base(no_prediction) e il punteggio sarà tra 0 e 1
   return (U_totale - U_no_predictions) / (U_optimal - U_no_predictions)

#Leggo il mio file 
file= pd.read_csv("sepsis3_hourly_labeled.csv")

#Converto sofa_time, suspected_infection_time e intime in formato datetime per poterli usare nei calcoli 
file["sofa_time"] = pd.to_datetime(file["sofa_time"], errors="coerce")
file["suspected_infection_time"] = pd.to_datetime(file["suspected_infection_time"], errors="coerce")
file["intime"] = pd.to_datetime(file["intime"], errors="coerce")

#L'onset della sepsi è il minimo tra sofa_time e suspected_infection_time, come definito nel paper
file["sepsis_onset"] = file[["sofa_time","suspected_infection_time"]].min(axis=1)

#Calcolo a quante ore dall'ammissione in ICU avviene la sepsi
file["sepsis_onset_hour"] = (file["sepsis_onset"] - file["intime"]).dt.total_seconds() / 3600

#Calcolo quante ore mancano alla sepsi per ogni riga (negativo = sepsi già avvenuta)
file["hours_to_sepsis"] = file["sepsis_onset_hour"] - file["hour_index"]

# split temporale basato su anchor_year_group — training sui ricoveri più vecchi, test sui più recenti
train_ids = file[file["anchor_year_group"].isin(["2008 - 2010", "2011 - 2013"])]["subject_id"].unique()
val_ids = file[file["anchor_year_group"] == "2014 - 2016"]["subject_id"].unique()
test_ids = file[file["anchor_year_group"] == "2017 - 2019"]["subject_id"].unique()

#Prendo le righe dei pazienti che appartengono al corrispettivo gruppo 
test_set=file[file["subject_id"].isin(test_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)
validation_set=file[file["subject_id"].isin(val_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)
train_set=file[file["subject_id"].isin(train_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)

#Tolgo le colonne identificative e la label dalla X, la Y contiene solo la label, servirà per confrontare le predizioni del modello con la realtà
X_train=train_set.drop(["subject_id","hadm_id","stay_id","label_sepsis_6h","Gender","hour_start","hour_end","intime","antibiotic_time","culture_time","suspected_infection_time","sofa_time","sepsis3","sepsis_onset","is_sepsis","sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2","TroponinI","anchor_year_group","respiration","coagulation","liver","cardiovascular","cns","renal","icu_hours"],axis=1)
Y_train=train_set["label_sepsis_6h"]

X_val=validation_set.drop(["subject_id","hadm_id","stay_id","label_sepsis_6h","Gender","hour_start","hour_end","intime","antibiotic_time","culture_time","suspected_infection_time","sofa_time","sepsis3","sepsis_onset","is_sepsis","sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2","TroponinI","anchor_year_group","respiration","coagulation","liver","cardiovascular","cns","renal","icu_hours"],axis=1)
Y_val=validation_set["label_sepsis_6h"]

X_test=test_set.drop(["subject_id","hadm_id","stay_id","label_sepsis_6h","Gender","hour_start","hour_end","intime","antibiotic_time","culture_time","suspected_infection_time","sofa_time","sepsis3","sepsis_onset","is_sepsis","sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2","TroponinI","anchor_year_group","respiration","coagulation","liver","cardiovascular","cns","renal","icu_hours"],axis=1)
Y_test=test_set["label_sepsis_6h"]

#Prendo i nomi delle colonne che usero come feature 
feature_cols = X_train.columns.tolist()

#Griglia di iperparametri per XGBOOST
param_grid_xgb = {
    "n_estimators": [100, 200, 300], #quanti alberi costruisce
    "max_depth": [3, 5, 7], #quanto profondi possono essere gli alberi
    "learning_rate": [0.01, 0.1, 0.3]   #quanto veloce impara
}

#Gestisco i -1 e i NaN con forward fill per paziente,evito di mescolare dati tra pazienti diversi
X_train_ffill = train_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)
X_val_ffill = validation_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)
X_test_ffill = test_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)

# scaling per MLP e LSTM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ffill)
X_val_scaled = scaler.transform(X_val_ffill)
X_test_scaled = scaler.transform(X_test_ffill)
#Preparo e alleno il modello XGBoost
#unisco train e validation per passarli al GridSearch
X_train_val = pd.DataFrame(np.concatenate([X_train_ffill, X_val_ffill]), columns=feature_cols)
X_val_scaled_df = pd.DataFrame(X_val_ffill, columns=feature_cols)
Y_train_val = pd.concat([Y_train, Y_val]).reset_index(drop=True)

#Definisco gli indici per dire al GridSearch quali righe sono train e quali val
n_train = len(X_train_scaled)
n_val = len(X_val_scaled)
split = [(list(range(n_train)), list(range(n_train, n_train + n_val)))]

#Provo tutte le combinazioni della griglia e valuto sul validation set con AUROC,l'early stopping ferma il training se la performance non migliora per 10 round consecutivi
grid_xgb = GridSearchCV(
    XGBClassifier(early_stopping_rounds=10, eval_metric="auc"),
    param_grid_xgb, cv=split, scoring="roc_auc", n_jobs=1,
    error_score=0
)
grid_xgb.fit(X_train_val, Y_train_val, eval_set=[(X_val_scaled_df, Y_val)],verbose=False)

#Prendo il modello con i migliori parametri trovati
model = grid_xgb.best_estimator_
pred_xgb = model.predict(pd.DataFrame(X_val_ffill, columns=feature_cols)).flatten()
t_test_xgb = model.predict(pd.DataFrame(X_test_ffill, columns=feature_cols)).flatten()
print("Migliori parametri XGBoost:", grid_xgb.best_params_)

# creo un dataframe con i dati scalati mantenendo subject_id, hour_index e label
train_scaled_df = train_set[["subject_id", "hour_index", "label_sepsis_6h"]].copy().reset_index(drop=True)
train_scaled_df[feature_cols] = X_train_scaled

val_scaled_df = validation_set[["subject_id", "hour_index", "label_sepsis_6h"]].copy().reset_index(drop=True)
val_scaled_df[feature_cols] = X_val_scaled

test_scaled_df = test_set[["subject_id", "hour_index", "label_sepsis_6h"]].copy().reset_index(drop=True)
test_scaled_df[feature_cols] = X_test_scaled

#Creo le sequenze temporali di 12 ore per ogni paziente nei tre set
X_train_seq, Y_train_seq = crea_sequenze(train_scaled_df, feature_cols, 24)
X_val_seq, Y_val_seq = crea_sequenze(val_scaled_df, feature_cols, 24)
X_test_seq, Y_test_seq = crea_sequenze(test_scaled_df, feature_cols, 24)

#Tuning manuale MLP (perchè mi dava problemi)
best_auroc_mlp = 0
best_params_mlp = {}
mlp = None
random.seed(42)
combinazioni_mlp = [(u, e, b) for u in [64, 128, 256] for e in [10, 30, 50] for b in [128, 256]]
combinazioni_scelte_mlp = random.sample(combinazioni_mlp, 10)
for units, epochs, batch_size in combinazioni_scelte_mlp:
    mlp_temp = Sequential()
    mlp_temp.add(Dense(units=units, activation="relu", input_dim=X_train_scaled.shape[1]))
    mlp_temp.add(Dense(units=32, activation="relu"))
    mlp_temp.add(Dense(units=1, activation="sigmoid"))
    mlp_temp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True) #early stopping ferma il training se la val_loss non migliora per 3 epoch consecutive e ripristina i pesi migliori trovati durante il training
    mlp_temp.fit(X_train_scaled, Y_train, epochs=epochs, batch_size=batch_size, verbose=0,
    validation_data=(X_val_scaled, Y_val), callbacks=[early_stop])
            
    prob = mlp_temp.predict(X_val_scaled,verbose = 0).flatten()
    auroc = roc_auc_score(Y_val, prob)
           
    if auroc > best_auroc_mlp:
        best_auroc_mlp = auroc
        best_params_mlp = {"units": units, "epochs": epochs, "batch_size": batch_size}
        mlp = mlp_temp
if mlp is None:
    print("Errore: nessun modello MLP trovato")
else:
    t_mlp = (mlp.predict(X_val_scaled,verbose = 0) > 0.5).astype(int).flatten()
    t_test_mlp = (mlp.predict(X_test_scaled,verbose = 0) > 0.5).astype(int).flatten()
    print("Migliori parametri MLP:", best_params_mlp)

#Preparo la LSTM

#Tuning manuale LSTM (perchè dava errore)
best_auroc_lstm = 0
best_params_lstm = {}
lstm = None
combinazioni_lstm = [(u, e, b) for u in [64, 128] for e in [10, 30, 50] for b in [128, 256]]
combinazioni_scelte_lstm = random.sample(combinazioni_lstm, 10)

for units, epochs, batch_size in combinazioni_scelte_lstm:
    lstm_temp = Sequential()
    lstm_temp.add(LSTM(units=units, input_shape=(24, len(feature_cols))))
    lstm_temp.add(Dense(units=1, activation="sigmoid"))
    lstm_temp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True) #early stopping ferma il training se la val_loss non migliora per 3 epoch consecutive e ripristina i pesi migliori trovati durante il training
    lstm_temp.fit(X_train_seq, Y_train_seq, epochs=epochs, batch_size=batch_size, verbose=0,
    validation_data=(X_val_seq, Y_val_seq), callbacks=[early_stop])
            
    prob = lstm_temp.predict(X_val_seq,verbose=0).flatten()
    auroc = roc_auc_score(Y_val_seq, prob)
            
    if auroc > best_auroc_lstm:
        best_auroc_lstm = auroc
        best_params_lstm = {"units": units, "epochs": epochs, "batch_size": batch_size}
        lstm = lstm_temp
if lstm is None:
    print("Errore: nessun modello LSTM trovato")
else:
    t_lstm = (lstm.predict(X_val_seq,verbose = 0) > 0.5).astype(int).flatten()
    t_test_lstm = (lstm.predict(X_test_seq,verbose=0) > 0.5).astype(int).flatten()
    print("Migliori parametri LSTM:", best_params_lstm)

#Risultati del Validation Set
print("\n ---------- Validation Set ----------")
print("\nXGBOOST: ")
#Calcolo la probabilità per AUROC e AUPRC
t_xgb_prob = model.predict_proba(pd.DataFrame(X_val_ffill, columns=feature_cols))[:, 1]
evaluetion_metrics(Y_val,pred_xgb,t_xgb_prob)
#Calcolo l'utilità clinica normalizzata, passo le ore mancanti alla sepsi, se il paziente è settico e le predizioni del modello t sono le predizioni di XGBoost
punteggi_xgb = normalizza_punteggio(validation_set["hours_to_sepsis"], validation_set["is_sepsis"], pred_xgb)
print("Utilità clinica normalizzata XGBoost:", punteggi_xgb)

print("\nMLP: ")
#Calcolo la probabilità per AUROC e AUPRC,il predict restituisce un array [N,1] ma lo score lo vuole "piatto"->[N]
t_mlp_prob = mlp.predict(X_val_scaled,verbose = 0).flatten()
evaluetion_metrics(Y_val,t_mlp,t_mlp_prob)
punteggi_mlp = normalizza_punteggio(validation_set["hours_to_sepsis"], validation_set["is_sepsis"],t_mlp)
print("Media utilità clinica MLP:", punteggi_mlp)

print("\nLSTM: ")
#Calcolo la probabilità per AUROC e AUPRC,il predict restituisce un array [N,1] ma lo score lo vuole "piatto"->[N]
t_lstm_prob = lstm.predict(X_val_seq,verbose=0).flatten()
evaluetion_metrics(Y_val_seq,t_lstm,t_lstm_prob)

#La LSTM lavora su sequenze per paziente quindi ho una sola predizione e non una per ogni singola ora 
punteggi_lstm = normalizza_punteggio(validation_set["hours_to_sepsis"], validation_set["is_sepsis"], t_lstm)
print("Media utilità clinica LSTM:", punteggi_lstm)

print("\n ---------- Test Set ----------")
#Risultati del Test Set
print("\nXGBOOST Test Set: ")
t_xgb_test_prob = model.predict_proba(pd.DataFrame(X_test_ffill, columns=feature_cols))[:, 1]
evaluetion_metrics(Y_test, t_test_xgb, t_xgb_test_prob)
punteggi_xgb_test=normalizza_punteggio(test_set["hours_to_sepsis"], test_set["is_sepsis"],t_test_xgb)
print("Media utilità clinica XGBoost Test set:", punteggi_xgb_test)

print("\nMLP Test Set:")
t_mlp_prob = mlp.predict(X_test_scaled,verbose=0).flatten()
evaluetion_metrics(Y_test,t_test_mlp,t_mlp_prob)
punteggi_mlp_test=normalizza_punteggio(test_set["hours_to_sepsis"], test_set["is_sepsis"],t_test_mlp)
print("Media utilità clinica MLP Test set:", punteggi_mlp_test)

print("\nLSTM Test Set: ")
t_lstm_prob = lstm.predict(X_test_seq,verbose=0).flatten()
evaluetion_metrics(Y_test_seq, t_test_lstm,t_lstm_prob)
punteggi_lstm_test = normalizza_punteggio(test_set["hours_to_sepsis"], test_set["is_sepsis"], t_test_lstm)
print("Media utilità clinica LSTM Test set:", punteggi_lstm_test)