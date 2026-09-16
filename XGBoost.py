import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import shap
import matplotlib.pyplot as plt

#Confronto y_true(reale) con le y_pred del modello per vedere quanto "bravo" è il modello,y_prob è la probabilità,AUROC e AUPRC le richiedono perchè misurano quanto bene il modello ordina i pazienti dal più al meno rischioso 
def evaluetion_metrics(y_true,y_pred,y_prob):
    #quanto bene il modello distingue settici da non 
    auroc=roc_auc_score(y_true,y_prob)
    #misura precisione e sensibilità
    auprc=average_precision_score(y_true,y_prob)
    #quante predizioni sono corrette(sul totale)
    accuracy=accuracy_score(y_true,y_pred)
    #tra tutti quelli che il modello dice "sepsi", quanti lo sono davvero
    precision=precision_score(y_true,y_pred)
    #tra tutti i veri settici, quanti il modello riesce a beccare
    recall=recall_score(y_true,y_pred)
    #bilancia i falsi positivi e i falsi negativi 
    f1=f1_score(y_true,y_pred)
    print("\nAUROC",auroc,"\n\nAUPRC",auprc,"\n\nAccuracy",accuracy,"\n\nPrecision",precision,"\n\nRecall",recall,"\n\nF1 Score",f1,"\n")
    return{"AUROC":auroc,"AUPRC":auprc,"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1 Score":f1}

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

# Il vero punteggio ottimale per riga è il massimo tra predire 1 e predire 0, non sempre 1
# predire sempre 1 è subottimale sulle righe non settiche e su quelle troppo lontane dall'onset,
# dove predire 0 costa 0 invece di -0.05 altrimente gonfiava l'utilità normalizzata
def punteggio_ottimale(hours_to_sepsis, is_sepsis):
    return max(
        calcola_punteggio(hours_to_sepsis, 1, is_sepsis),
        calcola_punteggio(hours_to_sepsis, 0, is_sepsis)
    )

#Sto normalizzando i punteggicome nel paper
def normalizza_punteggio(hours_to_sepsis_list,is_sepsis_list,prediction):
   #è il punteggio reale del modello con le sue predizioni
   U_totale = sum([calcola_punteggio(ore, pred, sepsi) for ore, pred, sepsi in zip(hours_to_sepsis_list, prediction, is_sepsis_list)])

   #questo è il punteggio nel caso peggiore in cui non predice mai sepsi(cioè il punto di partenza)
   U_no_predictions=sum([calcola_punteggio(ore, 0, sepsi) for ore, sepsi in zip(hours_to_sepsis_list, is_sepsis_list)])

   #questo è il punteggio se il modello predice sempre sepsi (cioè il massimo che posso avere)
   U_optimal=sum([punteggio_ottimale(ore,sepsi) for ore,sepsi in zip(hours_to_sepsis_list, is_sepsis_list)])

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

# NB: includo anche label_infection_6h e label_organ_6h tra le colonne da escludere,
# altrimenti restano come feature e causano leakage (stesso errore corretto in MLP/LSTM)
colonne_da_escludere = ["subject_id","hadm_id","stay_id","label_sepsis_6h","label_infection_6h","label_organ_6h","Gender","hour_start","hour_end","intime","antibiotic_time","culture_time","suspected_infection_time","sofa_time","sepsis3","sepsis_onset","is_sepsis","sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2","TroponinI","EtCO2","SaO2","anchor_year_group","anchor_age","anchor_year","hour_index","respiration","coagulation","liver","cardiovascular","cns","renal","icu_hours"]
X_train=train_set.drop(colonne_da_escludere,axis=1)
Y_train=train_set["label_sepsis_6h"]

X_val=validation_set.drop(colonne_da_escludere,axis=1)
Y_val=validation_set["label_sepsis_6h"]

X_test=test_set.drop(colonne_da_escludere,axis=1)
Y_test=test_set["label_sepsis_6h"]

#Prendo i nomi delle colonne che usero come feature 
feature_cols = X_train.columns.tolist()

#Griglia di iperparametri per XGBOOST
param_grid_xgb = {
    "n_estimators": [100, 200, 300], #quanti alberi costruisce
    "max_depth": [3, 5, 7], #quanto profondi possono essere gli alberi
    "learning_rate": [0.01, 0.1, 0.3] ,  #quanto veloce impara
    "subsample": [0.7, 0.8, 1.0], #righe campionate per ogni albero
    "colsample_bytree": [0.7, 0.8, 1.0] #feature campionate per ogni albero
}

#Gestisco i -1 e i NaN con forward fill per paziente,evito di mescolare dati tra pazienti diversi
X_train_ffill = train_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)
X_val_ffill = validation_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)
X_test_ffill = test_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)

#Preparo e alleno il modello XGBoost
#unisco train e validation per passarli al GridSearch
X_train_val = pd.DataFrame(np.concatenate([X_train_ffill, X_val_ffill]), columns=feature_cols)
X_val_ffill_df = pd.DataFrame(X_val_ffill, columns=feature_cols)
X_test_ffill_df = pd.DataFrame(X_test_ffill, columns=feature_cols) 
Y_train_val = pd.concat([Y_train, Y_val]).reset_index(drop=True)

#Definisco gli indici per dire al GridSearch quali righe sono train e quali val
n_train = len(X_train_ffill)
n_val = len(X_val_ffill)
split = [(list(range(n_train)), list(range(n_train, n_train + n_val)))]

#Faccio ripetere il tuning, valutazione e SHAP su piu seed
seeds = [42, 123, 7, 2024, 99]
risultati_multi_run = {"AUROC": [], "AUPRC": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [], "Utilita": []}

for seed_run in seeds:
    print(f"\n\n--------- SEED {seed_run} ---------")

    grid_xgb = RandomizedSearchCV(
        XGBClassifier(early_stopping_rounds=10, eval_metric="auc", random_state=seed_run),
        param_grid_xgb, n_iter=10, cv=split, scoring="roc_auc", n_jobs=1,
        error_score=0, random_state=seed_run
    )
    grid_xgb.fit(X_train_val, Y_train_val, eval_set=[(X_val_ffill_df, Y_val)], verbose=False)

    model = grid_xgb.best_estimator_
    print(f"Migliori parametri XGBoost seed {seed_run}:", grid_xgb.best_params_)

    t_test_xgb = model.predict(X_test_ffill_df).flatten()
    t_xgb_test_prob = model.predict_proba(X_test_ffill_df)[:, 1]

    print(f"\n--- Risultati Test Set (seed {seed_run}) ---")
    metriche = evaluetion_metrics(Y_test, t_test_xgb, t_xgb_test_prob)
    punteggi_xgb_test = normalizza_punteggio(test_set["hours_to_sepsis"], test_set["is_sepsis"], t_test_xgb)
    print("Media utilita' clinica XGBoost Test set:", punteggi_xgb_test)

    for chiave in ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1 Score"]:
        risultati_multi_run[chiave].append(metriche[chiave])
    risultati_multi_run["Utilita"].append(punteggi_xgb_test)

    explainer = shap.TreeExplainer(model)
    X_test_sample = X_test_ffill_df.sample(n=2000, random_state=42)
    shap_values = explainer.shap_values(X_test_sample)

    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"XGB_Plot_feature_{seed_run}.png", dpi=150)
    plt.close()

    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"XGB_Plot_completo_{seed_run}.png", dpi=150)
    plt.close()

#Qua aggiungo media +/- deviazione standard su tutti i seed
print(f"\n\n========== RISULTATI FINALI XGBOOST: MEDIA +/- DEV. STANDARD SU {len(seeds)} SEED ==========")
for chiave, valori in risultati_multi_run.items():
    media = np.mean(valori)
    std = np.std(valori)
    print(f"{chiave}: {media:.4f} +/- {std:.4f}")