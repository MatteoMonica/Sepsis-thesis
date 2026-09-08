import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
rng = random.Random(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Sto calcolando i punteggi da dare come nel paper (PhysioNet)
# NB: la finestra di reward/penalità per pazienti settici si estende da 12h prima dell'onset
# fino a 3h DOPO l'onset (non si azzera esattamente a hours_to_sepsis=0)
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

# Sto normalizzando i punteggicome nel paper
def normalizza_punteggio(hours_to_sepsis_list,is_sepsis_list,prediction):
   U_totale = sum([calcola_punteggio(ore, pred, sepsi) for ore, pred, sepsi in zip(hours_to_sepsis_list, prediction, is_sepsis_list)])
   U_no_predictions=sum([calcola_punteggio(ore, 0, sepsi) for ore, sepsi in zip(hours_to_sepsis_list, is_sepsis_list)])
   U_optimal=sum([calcola_punteggio(ore,1,sepsi) for ore,sepsi in zip(hours_to_sepsis_list, is_sepsis_list)])
   return (U_totale - U_no_predictions) / (U_optimal - U_no_predictions)

# Leggo il mio file 
file= pd.read_csv("sepsis3_hourly_labeled.csv")

#Converto sofa_time, suspected_infection_time e intime in formato datetime per poterli usare nei calcoli 
file["sofa_time"] = pd.to_datetime(file["sofa_time"], errors="coerce")
file["suspected_infection_time"] = pd.to_datetime(file["suspected_infection_time"], errors="coerce")
file["intime"] = pd.to_datetime(file["intime"], errors="coerce")

# L'onset della sepsi è il minimo tra sofa_time e suspected_infection_time, come definito nel paper
file["sepsis_onset"] = file[["sofa_time","suspected_infection_time"]].min(axis=1)

# Calcolo a quante ore dall'ammissione in ICU avviene la sepsi
file["sepsis_onset_hour"] = (file["sepsis_onset"] - file["intime"]).dt.total_seconds() / 3600

# Calcolo quante ore mancano alla sepsi per ogni riga (negativo = sepsi già avvenuta)
file["hours_to_sepsis"] = file["sepsis_onset_hour"] - file["hour_index"]

# split temporale basato su anchor_year_group — training sui ricoveri più vecchi, test sui più recenti
train_ids = file[file["anchor_year_group"].isin(["2008 - 2010", "2011 - 2013"])]["subject_id"].unique()
val_ids = file[file["anchor_year_group"] == "2014 - 2016"]["subject_id"].unique()
test_ids = file[file["anchor_year_group"] == "2017 - 2019"]["subject_id"].unique()

# Prendo le righe dei pazienti che appartengono al corrispettivo gruppo 
test_set=file[file["subject_id"].isin(test_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)
validation_set=file[file["subject_id"].isin(val_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)
train_set=file[file["subject_id"].isin(train_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)

# Tolgo le colonne identificative e la label dalla X (incluse quelle anti-leakage SOFA)
colonne_da_escludere = ["subject_id","hadm_id","stay_id","label_sepsis_6h","label_infection_6h","label_organ_6h","Gender","hour_start","hour_end","intime","antibiotic_time","culture_time","suspected_infection_time","sofa_time","sepsis3","sepsis_onset","is_sepsis","sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2","TroponinI","anchor_year_group","respiration","coagulation","liver","cardiovascular","cns","renal","icu_hours"]

X_train=train_set.drop(colonne_da_escludere,axis=1)
Y_train=train_set["label_sepsis_6h"]

X_val=validation_set.drop(colonne_da_escludere,axis=1)
Y_val=validation_set["label_sepsis_6h"]

X_test=test_set.drop(colonne_da_escludere,axis=1)
Y_test=test_set["label_sepsis_6h"]

# Prendo i nomi delle colonne che usero come feature 
feature_cols = X_train.columns.tolist()

# Gestisco i -1 e i NaN con forward fill per paziente,evito di mescolare dati tra pazienti diversi
X_train_ffill = train_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)
X_val_ffill = validation_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)
X_test_ffill = test_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)

# scaling per MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ffill)
X_val_scaled = scaler.transform(X_val_ffill)
X_test_scaled = scaler.transform(X_test_ffill)

# converto tutto in tensori PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.float32).view(-1, 1)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

print("Numero feature:", len(feature_cols))
print("Shape X_train_tensor:", X_train_tensor.shape)


# stessa architettura Dense(units)->Dense(32)->Dense(1,sigmoid) che avevi in Keras,
# tradotta in PyTorch: nn.Linear + ReLU al posto di Dense(activation="relu"),
# sigmoid finale applicata nel forward come per il multitask
class SepsisMLP(nn.Module):
    def __init__(self, n_features, units):
        super().__init__()
        self.rete = nn.Sequential(
            nn.Linear(n_features, units),
            nn.ReLU(),
            nn.Linear(units, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        logit = self.rete(x)
        p = torch.sigmoid(logit)
        return p


# Tuning manuale MLP 
# rispetto alla versione Keras, qui "epochs" non è più un iperparametro da provare:
# lo decide l'early stopping (n_epoche_max + pazienza), stesso approccio del multitask.
# tuning quindi su units, lr, batch_size
combinazioni_mlp = [(units, lr, batch_size)
                     for units in [64, 128, 256]
                     for lr in [0.001, 0.0003, 0.0001]
                     for batch_size in [128, 256]]
combinazioni_scelte_mlp = rng.sample(combinazioni_mlp, 10)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
X_val_device = X_val_tensor.to(device)

best_auroc_mlp = 0.0
best_params_mlp = {}
migliori_pesi_mlp = None

for units, lr, batch_size in combinazioni_scelte_mlp:
    print(f"\n=== Provo units={units}, lr={lr}, batch_size={batch_size} ===")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    mlp_temp = SepsisMLP(n_features=len(feature_cols), units=units).to(device)
    optimizer = torch.optim.Adam(mlp_temp.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    miglior_auroc_combo = 0.0
    pazienza = 3
    epoche_senza_miglioramento = 0
    pesi_migliori_combo = None
    n_epoche_max = 30

    for epoca in range(n_epoche_max):
        mlp_temp.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            p = mlp_temp(x_batch)
            loss = loss_fn(p, y_batch)
            loss.backward()
            optimizer.step()

        mlp_temp.eval()
        with torch.no_grad():
            p_val = mlp_temp(X_val_device).cpu().numpy()
        auroc_val_epoca = roc_auc_score(Y_val, p_val)

        if auroc_val_epoca > miglior_auroc_combo:
            miglior_auroc_combo = auroc_val_epoca
            epoche_senza_miglioramento = 0
            pesi_migliori_combo = {k: v.clone() for k, v in mlp_temp.state_dict().items()}
        else:
            epoche_senza_miglioramento += 1

        if epoche_senza_miglioramento >= pazienza:
            break

    print(f"Migliore AUROC per questa combinazione: {miglior_auroc_combo:.4f} (fermato all'epoca {epoca+1})")

    if miglior_auroc_combo > best_auroc_mlp:
        best_auroc_mlp = miglior_auroc_combo
        best_params_mlp = {"units": units, "lr": lr, "batch_size": batch_size}
        migliori_pesi_mlp = pesi_migliori_combo

print("\nMigliori parametri MLP:", best_params_mlp)

# ricostruisco il modello finale con i migliori iperparametri e i pesi migliori trovati
mlp = SepsisMLP(n_features=len(feature_cols), units=best_params_mlp["units"]).to(device)
mlp.load_state_dict(migliori_pesi_mlp)
mlp.eval()

with torch.no_grad():
    t_train_prob = mlp(X_train_tensor.to(device)).cpu().numpy().flatten()
t_train = (t_train_prob > 0.5).astype(int)

print("\n--- Risultati Train Set (confronto overfitting) ---")
evaluetion_metrics(Y_train, t_train, t_train_prob)
punteggi_train = normalizza_punteggio(train_set["hours_to_sepsis"], train_set["is_sepsis"], t_train)
print("Media utilità clinica MLP Train set:", punteggi_train)

# Predizioni su validation e test 
with torch.no_grad():
    t_mlp_prob = mlp(X_val_device).cpu().numpy().flatten()
    t_test_prob = mlp(X_test_tensor.to(device)).cpu().numpy().flatten()

t_mlp = (t_mlp_prob > 0.5).astype(int)
t_test_mlp = (t_test_prob > 0.5).astype(int)

# Risultati del Validation Set
print("\n ---------- Validation Set ----------")
print("\nMLP: ")
evaluetion_metrics(Y_val, t_mlp, t_mlp_prob)
punteggi_mlp = normalizza_punteggio(validation_set["hours_to_sepsis"], validation_set["is_sepsis"], t_mlp)
print("Media utilità clinica MLP:", punteggi_mlp)

print("\n ---------- Test Set ----------")
print("\nMLP Test Set:")
evaluetion_metrics(Y_test, t_test_mlp, t_test_prob)
punteggi_mlp_test = normalizza_punteggio(test_set["hours_to_sepsis"], test_set["is_sepsis"], t_test_mlp)
print("Media utilità clinica MLP Test set:", punteggi_mlp_test)