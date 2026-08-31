import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import random

torch.manual_seed(42)  # stessa pratica di riproducibilità usata per MLP/LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Leggo il CSV e ricostruisco le colonne temporali
file = pd.read_csv("sepsis3_hourly_labeled.csv")
file["sofa_time"] = pd.to_datetime(file["sofa_time"], errors="coerce")
file["suspected_infection_time"] = pd.to_datetime(file["suspected_infection_time"], errors="coerce")
file["intime"] = pd.to_datetime(file["intime"], errors="coerce")
file["sepsis_onset"] = file[["sofa_time","suspected_infection_time"]].min(axis=1)
file["sepsis_onset_hour"] = (file["sepsis_onset"] - file["intime"]).dt.total_seconds() / 3600
file["hours_to_sepsis"] = file["sepsis_onset_hour"] - file["hour_index"]

# Stesso split temporale 
train_ids = file[file["anchor_year_group"].isin(["2008 - 2010", "2011 - 2013"])]["subject_id"].unique()
val_ids = file[file["anchor_year_group"] == "2014 - 2016"]["subject_id"].unique()
test_ids = file[file["anchor_year_group"] == "2017 - 2019"]["subject_id"].unique()

train_set = file[file["subject_id"].isin(train_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)
val_set = file[file["subject_id"].isin(val_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)
test_set = file[file["subject_id"].isin(test_ids)].sort_values(["subject_id","hour_index"]).reset_index(drop=True)

# Stesse colonne da escludere ( incluse quelle anti-leakage) 
colonne_da_escludere = [
    "subject_id","hadm_id","stay_id","label_sepsis_6h","label_infection_6h","label_organ_6h",
    "Gender","hour_start","hour_end","intime","antibiotic_time","culture_time",
    "suspected_infection_time","sofa_time","sepsis3","sepsis_onset","is_sepsis",
    "sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2","TroponinI",
    "anchor_year_group","respiration","coagulation","liver","cardiovascular","cns","renal","icu_hours"
]

X_train = train_set.drop(colonne_da_escludere, axis=1)
X_val = val_set.drop(colonne_da_escludere, axis=1)
X_test = test_set.drop(colonne_da_escludere, axis=1)

feature_cols = X_train.columns.tolist()

#  I 3 target per ciascuno split 
Y_train_sepsi = train_set["label_sepsis_6h"]
Y_train_inf = train_set["label_infection_6h"]
Y_train_org = train_set["label_organ_6h"]

Y_val_sepsi = val_set["label_sepsis_6h"]
Y_val_inf = val_set["label_infection_6h"]
Y_val_org = val_set["label_organ_6h"]

Y_test_sepsi = test_set["label_sepsis_6h"]
Y_test_inf = test_set["label_infection_6h"]
Y_test_org = test_set["label_organ_6h"]

# Forward-fill per paziente + scaling
X_train_ffill = train_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)
X_val_ffill = val_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)
X_test_ffill = test_set.groupby("subject_id")[feature_cols].apply(lambda x: x.replace(-1, np.nan).ffill().fillna(0)).reset_index(drop=True)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ffill)
X_val_scaled = scaler.transform(X_val_ffill)
X_test_scaled = scaler.transform(X_test_ffill)

# Converto tutto in tensori PyTorch 
# le feature sono già numpy array (output di StandardScaler), le converto direttamente
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# le label le rendo colonna (N,1) con .view(-1,1), per combaciare con l'output del modello
Y_train_sepsi_tensor = torch.tensor(Y_train_sepsi.values, dtype=torch.float32).view(-1, 1)
Y_train_inf_tensor = torch.tensor(Y_train_inf.values, dtype=torch.float32).view(-1, 1)
Y_train_org_tensor = torch.tensor(Y_train_org.values, dtype=torch.float32).view(-1, 1)

Y_val_sepsi_tensor = torch.tensor(Y_val_sepsi.values, dtype=torch.float32).view(-1, 1)
Y_val_inf_tensor = torch.tensor(Y_val_inf.values, dtype=torch.float32).view(-1, 1)
Y_val_org_tensor = torch.tensor(Y_val_org.values, dtype=torch.float32).view(-1, 1)

Y_test_sepsi_tensor = torch.tensor(Y_test_sepsi.values, dtype=torch.float32).view(-1, 1)
Y_test_inf_tensor = torch.tensor(Y_test_inf.values, dtype=torch.float32).view(-1, 1)
Y_test_org_tensor = torch.tensor(Y_test_org.values, dtype=torch.float32).view(-1, 1)

print("Numero feature:", len(feature_cols))
print("Shape X_train_tensor:", X_train_tensor.shape)
print("Shape Y_train_sepsi_tensor:", Y_train_sepsi_tensor.shape)

class SepsisMultitaskParallelo(nn.Module):
    def __init__(self, n_features, dim_z=64):
        super().__init__()
        
        # Prende in input i dati clinici orari del paziente (es. HR, Temp, Lactate) 
        # e li comprime in un vettore "riassuntivo" z (dim_z) che ne racchiude le informazioni utili.
        self.encoder = nn.Sequential(
            # Primo strato: combinazione pesata + bias (n_features -> 128)
            nn.Linear(n_features, 128), 
            nn.ReLU(),                  # Attivazione: azzera i negativi per introdurre non-linearità
            
            # Secondo strato: mappa le feature estratte nello spazio latente z (128 -> dim_z)
            nn.Linear(128, dim_z),      
            nn.ReLU(),
        )
        
        # Riduce il vettore delle feature (dim_z) a un singolo valore lineare per il calcolo della probabilità
        self.infezione = nn.Linear(dim_z, 1)
        
        # Ramo parallelo per l'organo: mappa lo stesso vettore z a un singolo valore d'uscita
        self.organo = nn.Linear(dim_z, 1)
        
        # sepsis_head: prende z + P_infezione + P_organo + (P_infezione * P_organo)
        # quindi il suo input ha dimensione dim_z + 3
        self.sepsis_head = nn.Linear(dim_z + 3, 1)

    def forward(self, x):
        # Passa i dati grezzi del paziente attraverso la rete iniziale per estrarre le feature utili z
        # x ha forma (pazienti, parametri_clinici) -> z diventa (pazienti, 64)
        z = self.encoder(x) 
        
        # Calcola i punteggi grezzi (logit) per le due sotto-attività indipendenti
        # Esce un numero reale per ogni paziente (positivo o negativo)
        logit_inf = self.infezione(z)
        logit_org = self.organo(z)
        
        # Trasforma i logit grezzi in probabilità reali comprese tra 0 e 1 usando la Sigmoide
        p_inf = torch.sigmoid(logit_inf)
        p_org = torch.sigmoid(logit_org)
        
        # Costruisco l'input per la sepsis_head: prepara tutti gli elementi necessari
        # (contesto z + singoli rischi + rischio combinato) affinché il modello possa diagnosticare la sepsi
        input_sepsis = torch.cat([z, p_inf, p_org, p_inf * p_org], dim=1)

        # Calcola il logit finale per la sepsi e trasformalo nell'ultima probabilità
        logit_sepsis = self.sepsis_head(input_sepsis)
        p_sepsis = torch.sigmoid(logit_sepsis)
        
        # ritorno tutte e 3 le probabilità: mi servono per calcolare la loss combinata
        return p_sepsis, p_inf, p_org

# spazio di ricerca: le combinazioni di iperparametri da provare
combinazioni_multitask = [(dim_z, lr, batch_size)
                           for dim_z in [32, 64, 128]
                           for lr in [0.001, 0.0003, 0.0001]
                           for batch_size in [128, 256, 512]]
combinazioni_scelte_multitask = random.sample(combinazioni_multitask, 8)

# il dataset di train non dipende dagli iperparametri, lo creo una volta sola fuori dal ciclo
train_dataset = TensorDataset(X_train_tensor, Y_train_sepsi_tensor, Y_train_inf_tensor, Y_train_org_tensor)

# sposto il validation sulla GPU una volta sola, lo riuso per ogni combinazione (evito trasferimenti ripetuti)
X_val_device = X_val_tensor.to(device)

miglior_auroc_globale = 0.0
migliori_iperparametri = None
migliori_pesi = None

for dim_z, lr, batch_size in combinazioni_scelte_multitask:
    print(f"\n=== Provo dim_z={dim_z}, lr={lr}, batch_size={batch_size} ===")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # creo un modello NUOVO per ogni combinazione (pesi casuali di partenza),
    # altrimenti riuserei pesi già allenati dalla combinazione precedente
    model_temp = SepsisMultitaskParallelo(n_features=len(feature_cols), dim_z=dim_z).to(device)
    optimizer = torch.optim.Adam(model_temp.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # early stopping per QUESTA combinazione specifica
    miglior_auroc_val_combo = 0.0
    pazienza = 3
    epoche_senza_miglioramento = 0
    pesi_migliori_combo = None
    n_epoche_max = 30

    for epoca in range(n_epoche_max):
        model_temp.train()
        for x_batch, y_sepsi_batch, y_inf_batch, y_org_batch in train_loader:
            x_batch = x_batch.to(device)
            y_sepsi_batch = y_sepsi_batch.to(device)
            y_inf_batch = y_inf_batch.to(device)
            y_org_batch = y_org_batch.to(device)

            optimizer.zero_grad()
            p_sepsis, p_inf, p_org = model_temp(x_batch)

            loss_sepsis = loss_fn(p_sepsis, y_sepsi_batch)
            loss_inf = loss_fn(p_inf, y_inf_batch)
            loss_org = loss_fn(p_org, y_org_batch)
            loss = loss_sepsis + 0.5 * loss_inf + 0.5 * loss_org

            loss.backward()
            optimizer.step()

        model_temp.eval()
        with torch.no_grad():
            p_sepsis_val, _, _ = model_temp(X_val_device)
            p_sepsis_val = p_sepsis_val.cpu().numpy()
        auroc_val_epoca = roc_auc_score(Y_val_sepsi, p_sepsis_val)

        if auroc_val_epoca > miglior_auroc_val_combo:
            miglior_auroc_val_combo = auroc_val_epoca
            epoche_senza_miglioramento = 0
            pesi_migliori_combo = {k: v.clone() for k, v in model_temp.state_dict().items()}
        else:
            epoche_senza_miglioramento += 1

        if epoche_senza_miglioramento >= pazienza:
            break

    print(f"Migliore AUROC sepsi per questa combinazione: {miglior_auroc_val_combo:.4f} (fermato all'epoca {epoca+1})")

    # confronto col migliore globale visto finora tra TUTTE le combinazioni provate
    if miglior_auroc_val_combo > miglior_auroc_globale:
        miglior_auroc_globale = miglior_auroc_val_combo
        migliori_iperparametri = (dim_z, lr, batch_size)
        migliori_pesi = pesi_migliori_combo

print("\n=== RISULTATO FINALE RICERCA IPERPARAMETRI ===")
print("Migliori iperparametri (dim_z, lr, batch_size):", migliori_iperparametri)
print("Miglior AUROC sepsi validation:", miglior_auroc_globale)

# ricostruisco il modello finale con i migliori iperparametri trovati, e ci carico i pesi migliori
dim_z_finale, lr_finale, batch_size_finale = migliori_iperparametri
model = SepsisMultitaskParallelo(n_features=len(feature_cols), dim_z=dim_z_finale).to(device)
model.load_state_dict(migliori_pesi)

# metto il modello in "modalità valutazione"
model.eval()

# torch.no_grad() disattiva il calcolo dei gradienti: durante la valutazione è
# uno spreco di memoria e tempo
with torch.no_grad():
    p_sepsis_val, p_inf_val, p_org_val = model(X_val_device)

    # riporto le predizioni dalla GPU alla CPU e le converto in numpy:
    # sklearn (che uso per calcolare l'AUROC) lavora con array numpy, non con tensori PyTorch/GPU
    p_sepsis_val = p_sepsis_val.cpu().numpy()
    p_inf_val = p_inf_val.cpu().numpy()
    p_org_val = p_org_val.cpu().numpy()

# calcolo l'AUROC per ciascuno dei 3 task, confrontando le probabilità predette
# con le vere label del validation set (stessa metrica che usavi per XGBoost/MLP/LSTM)
auroc_sepsis = roc_auc_score(Y_val_sepsi, p_sepsis_val)
auroc_inf = roc_auc_score(Y_val_inf, p_inf_val)
auroc_org = roc_auc_score(Y_val_org, p_org_val)

with torch.no_grad():
    X_train_device = X_train_tensor.to(device)
    p_sepsis_train, p_inf_train, p_org_train = model(X_train_device)

    p_sepsis_train = p_sepsis_train.cpu().numpy()
    p_inf_train = p_inf_train.cpu().numpy()
    p_org_train = p_org_train.cpu().numpy()

auroc_sepsis_train = roc_auc_score(Y_train_sepsi, p_sepsis_train)
auroc_inf_train = roc_auc_score(Y_train_inf, p_inf_train)
auroc_org_train = roc_auc_score(Y_train_org, p_org_train)

# --- Valutazione finale sul TEST SET ---
# uso il modello con i migliori iperparametri e pesi trovati durante il tuning
model.eval()
with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    p_sepsis_test, p_inf_test, p_org_test = model(X_test_device)

    p_sepsis_test = p_sepsis_test.cpu().numpy()
    p_inf_test = p_inf_test.cpu().numpy()
    p_org_test = p_org_test.cpu().numpy()

auroc_sepsis_test = roc_auc_score(Y_test_sepsi, p_sepsis_test)
auroc_inf_test = roc_auc_score(Y_test_inf, p_inf_test)
auroc_org_test = roc_auc_score(Y_test_org, p_org_test)

print("\n--- Risultati Test Set (valutazione finale) ---")
print("AUROC sepsi (test):", auroc_sepsis_test)
print("AUROC infezione (test):", auroc_inf_test)
print("AUROC organo (test):", auroc_org_test)

print("\n--- Risultati Train Set (confronto overfitting) ---")
print("AUROC sepsi (train):", auroc_sepsis_train, " vs validation:", auroc_sepsis)
print("AUROC infezione (train):", auroc_inf_train, " vs validation:", auroc_inf)
print("AUROC organo (train):", auroc_org_train, " vs validation:", auroc_org)