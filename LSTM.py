import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

torch.manual_seed(42)
rng = random.Random(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Creo la funzione per crearmi il dataset da dare alla LSTM perchè serve diverso rispetto a quello usato dagli altri modelli
def crea_sequenze(dataset,feature_cols,n_ore):
    X_sequenze = []
    Y_label = [] 
    for patient_id,group in dataset.groupby("subject_id"):
        group = group.sort_values("hour_index")
        for t in group["hour_index"]:  
            finestra = group[group["hour_index"] <= t]
            sequenza = finestra.tail(n_ore)[feature_cols]
            if len(sequenza) < n_ore:
                padding = np.zeros((n_ore - len(sequenza),len(feature_cols)))
                sequenza = np.concatenate([padding,sequenza.values],axis=0)
            else:
                sequenza = sequenza.values
            X_sequenze.append(sequenza)
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
    #tra tutti quelli che il modello dice "sepsi", quanti lo sono davvero
    precision=precision_score(y_true,y_pred)
    #tra tutti i veri settici, quanti il modello riesce a beccare
    recall=recall_score(y_true,y_pred)
    #bilancia i falsi positivi e i falsi negativi 
    f1=f1_score(y_true,y_pred)
    print("\nAUROC",auroc,"\n\nAUPRC",auprc,"\n\nAccuracy",accuracy,"\n\nPrecision",precision,"\n\nRecall",recall,"\n\nF1 Score",f1,"\n")
    return{"AUROC":auroc,"AUPRC":auprc,"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1 Score":f1}

def calcola_punteggio(hours_to_sepsis,prediction,is_sepsis):
    if is_sepsis == False:
        if prediction == 1:
            return -0.05
        else:
            return 0
    else:
        if pd.isna(hours_to_sepsis):
            return 0
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
        elif hours_to_sepsis>=-3 and hours_to_sepsis <6:
            if prediction == 1:
                return  (hours_to_sepsis+3) / 9
            else:
                return -2 *(6-hours_to_sepsis)/9
        elif hours_to_sepsis < -3:
            if prediction == 1:
                return 0
            else:
                return -2    

def normalizza_punteggio(hours_to_sepsis_list,is_sepsis_list,prediction):
   U_totale = sum([calcola_punteggio(ore, pred, sepsi) for ore, pred, sepsi in zip(hours_to_sepsis_list, prediction, is_sepsis_list)])
   U_no_predictions=sum([calcola_punteggio(ore, 0, sepsi) for ore, sepsi in zip(hours_to_sepsis_list, is_sepsis_list)])
   U_optimal=sum([calcola_punteggio(ore,1,sepsi) for ore,sepsi in zip(hours_to_sepsis_list, is_sepsis_list)])
   return (U_totale - U_no_predictions) / (U_optimal - U_no_predictions)

#Leggo il mio file 
file= pd.read_csv("sepsis3_hourly_labeled.csv")
file["sofa_time"] = pd.to_datetime(file["sofa_time"], errors="coerce")
file["suspected_infection_time"] = pd.to_datetime(file["suspected_infection_time"], errors="coerce")
file["intime"] = pd.to_datetime(file["intime"], errors="coerce")
file["sepsis_onset"] = file[["sofa_time","suspected_infection_time"]].min(axis=1)
file["sepsis_onset_hour"] = (file["sepsis_onset"] - file["intime"]).dt.total_seconds() / 3600
file["hours_to_sepsis"] = file["sepsis_onset_hour"] - file["hour_index"]

train_ids = file[file["anchor_year_group"].isin(["2008 - 2010", "2011 - 2013"])]["subject_id"].unique()
val_ids = file[file["anchor_year_group"] == "2014 - 2016"]["subject_id"].unique()
test_ids = file[file["anchor_year_group"] == "2017 - 2019"]["subject_id"].unique()

test_set=file[file["subject_id"].isin(test_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)
validation_set=file[file["subject_id"].isin(val_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)
train_set=file[file["subject_id"].isin(train_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)

# NB: includo anche label_infection_6h e label_organ_6h tra le colonne da escludere,
# altrimenti restano come feature e causano leakage (errore già corretto nell'MLP)
colonne_da_escludere = ["subject_id","hadm_id","stay_id","label_sepsis_6h","label_infection_6h","label_organ_6h","Gender","hour_start","hour_end","intime","antibiotic_time","culture_time","suspected_infection_time","sofa_time","sepsis3","sepsis_onset","is_sepsis","sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2","TroponinI","EtCO2","SaO2","anchor_year_group","anchor_age","anchor_year","hour_index","respiration","coagulation","liver","cardiovascular","cns","renal","icu_hours"]
X_train=train_set.drop(colonne_da_escludere,axis=1)
Y_train=train_set["label_sepsis_6h"]
X_val=validation_set.drop(colonne_da_escludere,axis=1)
Y_val=validation_set["label_sepsis_6h"]
X_test=test_set.drop(colonne_da_escludere,axis=1)
Y_test=test_set["label_sepsis_6h"]

feature_cols = X_train.columns.tolist()

# Forward-fill + scaling (identico agli altri modelli)
X_train_ffill = train_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)
X_val_ffill = validation_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)
X_test_ffill = test_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ffill)
X_val_scaled = scaler.transform(X_val_ffill)
X_test_scaled = scaler.transform(X_test_ffill)

# ricostruisco i dataframe scalati mantenendo subject_id/hour_index/label,
# necessari per crea_sequenze (che raggruppa per paziente e ordina per ora)
train_scaled_df = train_set[["subject_id", "hour_index", "label_sepsis_6h"]].copy().reset_index(drop=True)
train_scaled_df[feature_cols] = X_train_scaled

val_scaled_df = validation_set[["subject_id", "hour_index", "label_sepsis_6h"]].copy().reset_index(drop=True)
val_scaled_df[feature_cols] = X_val_scaled

test_scaled_df = test_set[["subject_id", "hour_index", "label_sepsis_6h"]].copy().reset_index(drop=True)
test_scaled_df[feature_cols] = X_test_scaled

# costruisco le sequenze temporali di 24 ore per ogni paziente, nei tre set
X_train_seq, Y_train_seq = crea_sequenze(train_scaled_df, feature_cols, 24)
X_val_seq, Y_val_seq = crea_sequenze(val_scaled_df, feature_cols, 24)
X_test_seq, Y_test_seq = crea_sequenze(test_scaled_df, feature_cols, 24)

# converto le sequenze in tensori PyTorch: forma (n_esempi, 24_ore, n_feature)
X_train_seq_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
X_val_seq_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
X_test_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32)

Y_train_seq_tensor = torch.tensor(Y_train_seq, dtype=torch.float32).view(-1, 1)
Y_val_seq_tensor = torch.tensor(Y_val_seq, dtype=torch.float32).view(-1, 1)
Y_test_seq_tensor = torch.tensor(Y_test_seq, dtype=torch.float32).view(-1, 1)

print("Numero feature:", len(feature_cols))
print("Shape X_train_seq_tensor:", X_train_seq_tensor.shape)


# valuta il modello su un dataset grande, un batch alla volta, per non saturare la memoria GPU
# (una LSTM su migliaia di sequenze intere insieme richiede troppa memoria in un colpo solo,
# a differenza dell'MLP dove un forward pass su tutto il validation set è leggero)
def predici_a_batch(model, X_tensor, batch_size=256):
    model.eval()
    predizioni = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            p_batch = model(batch).cpu().numpy()
            predizioni.append(p_batch)
    return np.concatenate(predizioni, axis=0)


# LSTM in PyTorch: nn.LSTM prende in input sequenze di forma (batch, timestep, feature)
# e restituisce, per ogni timestep, uno stato nascosto; a noi serve solo l'ultimo
# (riassume l'intera sequenza di 24 ore), che passiamo a uno strato finale per la predizione
class SepsisLSTM(nn.Module):
    def __init__(self, n_features, units):
        super().__init__()
        # batch_first=True dice a PyTorch che la forma dell'input è (batch, timestep, feature)
        # invece del default (timestep, batch, feature) - più intuitivo così
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=units, batch_first=True)
        # strato finale: dallo stato nascosto (units numeri) a 1 probabilità
        self.output = nn.Linear(units, 1)

    def forward(self, x):
        # lstm_out ha forma (batch, 24, units): lo stato nascosto ad OGNI ora
        # hidden ha forma (1, batch, units): SOLO l'ultimo stato nascosto (fine sequenza)
        lstm_out, (hidden, cell) = self.lstm(x)
        # prendo l'ultimo stato nascosto (riassunto delle 24 ore), tolgo la dimensione extra
        ultimo_stato = hidden.squeeze(0)  # da (1, batch, units) a (batch, units)
        logit = self.output(ultimo_stato)
        p = torch.sigmoid(logit)
        return p


# --- Tuning manuale LSTM ---
combinazioni_lstm = [(units, lr, batch_size)
                      for units in [64, 128]
                      for lr in [0.001, 0.0003, 0.0001]
                      for batch_size in [128, 256]]
combinazioni_scelte_lstm = rng.sample(combinazioni_lstm, 10)

train_dataset_seq = TensorDataset(X_train_seq_tensor, Y_train_seq_tensor)

best_auroc_lstm = 0.0
best_params_lstm = {}
migliori_pesi_lstm = None

for units, lr, batch_size in combinazioni_scelte_lstm:
    print(f"\n=== Provo units={units}, lr={lr}, batch_size={batch_size} ===")

    train_loader_seq = DataLoader(train_dataset_seq, batch_size=batch_size, shuffle=True)

    lstm_temp = SepsisLSTM(n_features=len(feature_cols), units=units).to(device)
    optimizer = torch.optim.Adam(lstm_temp.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    miglior_auroc_combo = 0.0
    pazienza = 3
    epoche_senza_miglioramento = 0
    pesi_migliori_combo = None
    n_epoche_max = 30

    for epoca in range(n_epoche_max):
        lstm_temp.train()
        for x_batch, y_batch in train_loader_seq:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            p = lstm_temp(x_batch)
            loss = loss_fn(p, y_batch)
            loss.backward()
            optimizer.step()

        # valutazione a batch, per evitare l'OutOfMemory visto passando tutto il validation insieme
        p_val = predici_a_batch(lstm_temp, X_val_seq_tensor, batch_size=batch_size)
        auroc_val_epoca = roc_auc_score(Y_val_seq, p_val)

        if auroc_val_epoca > miglior_auroc_combo:
            miglior_auroc_combo = auroc_val_epoca
            epoche_senza_miglioramento = 0
            pesi_migliori_combo = {k: v.clone() for k, v in lstm_temp.state_dict().items()}
        else:
            epoche_senza_miglioramento += 1

        if epoche_senza_miglioramento >= pazienza:
            break

    print(f"Migliore AUROC per questa combinazione: {miglior_auroc_combo:.4f} (fermato all'epoca {epoca+1})")

    if miglior_auroc_combo > best_auroc_lstm:
        best_auroc_lstm = miglior_auroc_combo
        best_params_lstm = {"units": units, "lr": lr, "batch_size": batch_size}
        migliori_pesi_lstm = pesi_migliori_combo

print("\nMigliori parametri LSTM:", best_params_lstm)

lstm = SepsisLSTM(n_features=len(feature_cols), units=best_params_lstm["units"]).to(device)
lstm.load_state_dict(migliori_pesi_lstm)
lstm.eval()

# predizioni finali, anche queste a batch per lo stesso motivo
t_lstm_prob = predici_a_batch(lstm, X_val_seq_tensor).flatten()
t_test_prob = predici_a_batch(lstm, X_test_seq_tensor).flatten()

t_lstm = (t_lstm_prob > 0.5).astype(int)
t_test_lstm = (t_test_prob > 0.5).astype(int)

#Risultati del Validation Set
print("\n ---------- Validation Set ----------")
print("\nLSTM: ")
evaluetion_metrics(Y_val_seq, t_lstm, t_lstm_prob)
# NB: per l'utilità clinica uso validation_set (non val_scaled_df), che ha hours_to_sepsis/is_sepsis;
# crea_sequenze produce una riga per ogni (subject_id, hour_index) nello stesso ordine di validation_set
punteggi_lstm = normalizza_punteggio(validation_set["hours_to_sepsis"], validation_set["is_sepsis"], t_lstm)
print("Media utilità clinica LSTM:", punteggi_lstm)

print("\n ---------- Test Set ----------")
print("\nLSTM Test Set:")
evaluetion_metrics(Y_test_seq, t_test_lstm, t_test_prob)
punteggi_lstm_test = normalizza_punteggio(test_set["hours_to_sepsis"], test_set["is_sepsis"], t_test_lstm)
print("Media utilità clinica LSTM Test set:", punteggi_lstm_test)

import shap
import matplotlib.pyplot as plt

# sposto il modello su CPU, alcune versioni di SHAP hanno problemi di compatibilità
# con operazioni LSTM su GPU
lstm_cpu = lstm.to("cpu")
lstm_cpu.eval()

# punto di riferimento per calcolare quanto ogni feature si scosta dalla norma
background = X_train_seq_tensor[:100].to("cpu")

# campiono il test set, con le sequenze, GradientExplainer è più lento di
# TreeExplainer, quindi uso un campione più piccolo 
rng_shap = np.random.RandomState(42)
test_sample_idx = rng_shap.choice(len(X_test_seq_tensor), size=200, replace=False)
test_sample = X_test_seq_tensor[test_sample_idx].to("cpu")

explainer = shap.GradientExplainer(lstm_cpu, background)
shap_values = explainer.shap_values(test_sample)

# shap_values ha forma (200, 24, 34), un valore per ogni (esempio, ora, feature).
# Se viene restituito come lista (un elemento per output del modello), prendo il primo
if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.array(shap_values).reshape(200, 24, len(feature_cols))

# Aggregazione 1: importanza per FEATURE, sommando il valore assoluto su tutte le 24 ore
# così ottengo un singolo numero per, confrontabile con il caso XGBoost
importanza_per_feature = np.abs(shap_values).sum(axis=1)  
importanza_media = importanza_per_feature.mean(axis=0)     

# ordino le feature dalla più alla meno importante
ordine = np.argsort(importanza_media)[::-1]
feature_ordinate = [feature_cols[i] for i in ordine]
valori_ordinati = importanza_media[ordine]

plt.figure(figsize=(8, 10))
plt.barh(feature_ordinate[:20][::-1], valori_ordinati[:20][::-1])
plt.xlabel("Importanza media (|valore SHAP| sommato sulle 24 ore)")
plt.title("LSTM - Importanza globale delle feature (top 20)")
plt.tight_layout()
plt.savefig("LSTM_Plot_globale.png", dpi=150)
plt.close()

# Aggregazione 2: importanza per ORA, per vedere se le ore più recenti
# contano di più di quelle più lontane nel tempo (plausibile clinicamente) 
importanza_per_ora = np.abs(shap_values).mean(axis=(0, 2))  

plt.figure(figsize=(8, 4))
plt.plot(range(1, 25), importanza_per_ora, marker="o")
plt.xlabel("Ora nella sequenza (24 = ora più recente)")
plt.ylabel("Importanza media |SHAP|")
plt.title("LSTM - Importanza media per ora della sequenza")
plt.tight_layout()
plt.savefig("LSTM_Plot_temporale.png", dpi=150)
plt.close()

print("\nGrafici SHAP LSTM salvati: shap_lstm_importanza_globale.png, shap_lstm_importanza_temporale.png")