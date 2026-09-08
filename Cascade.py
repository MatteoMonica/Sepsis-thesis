import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import random

torch.manual_seed(42)
rng = random.Random(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Sto calcolando i punteggi da dare come nel paper (PhysioNet)
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

# Stesse colonne da escludere (incluse quelle anti-leakage)
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

# I 3 target per ciascuno split
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
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

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


# CASCADE: a differenza del Parallelo, sepsis_head riceve in input non solo z,
# ma anche le probabilità p_inf, p_org e il loro prodotto (z + 3 valori extra).
# Come da formula: p_s = SepsisHead(z, p_i, p_o, p_i*p_o)
# InfectionHead e OrganHead restano invece indipendenti tra loro, ricevono solo z
class SepsisMultitaskCascade(nn.Module):
    def __init__(self, n_features, dim_z=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, dim_z),
            nn.ReLU(),
        )
        self.infezione = nn.Linear(dim_z, 1)
        self.organo = nn.Linear(dim_z, 1)
        # input allargato: dim_z (z) + 1 (p_inf) + 1 (p_org) + 1 (p_inf*p_org) = dim_z + 3
        self.sepsis_head = nn.Linear(dim_z + 3, 1)

    def forward(self, x):
        z = self.encoder(x)

        logit_inf = self.infezione(z)
        logit_org = self.organo(z)
        p_inf = torch.sigmoid(logit_inf)
        p_org = torch.sigmoid(logit_org)

        # costruisco l'input della sepsis_head concatenando z con le predizioni
        # degli altri due sotto-task e la loro interazione (prodotto)
        input_sepsis = torch.cat([z, p_inf, p_org, p_inf * p_org], dim=1)
        logit_sepsis = self.sepsis_head(input_sepsis)
        p_sepsis = torch.sigmoid(logit_sepsis)

        return p_sepsis, p_inf, p_org


# Ricerca iperparametri (dim_z, lr, batch_size) con early stopping 
combinazioni_multitask = [(dim_z, lr, batch_size)
                           for dim_z in [32, 64, 128]
                           for lr in [0.001, 0.0003, 0.0001]
                           for batch_size in [128, 256, 512]]
combinazioni_scelte_multitask = rng.sample(combinazioni_multitask, 8)

train_dataset = TensorDataset(X_train_tensor, Y_train_sepsi_tensor, Y_train_inf_tensor, Y_train_org_tensor)
X_val_device = X_val_tensor.to(device)

miglior_auroc_globale = 0.0
migliori_iperparametri = None
migliori_pesi = None

for dim_z, lr, batch_size in combinazioni_scelte_multitask:
    print(f"\n=== Provo dim_z={dim_z}, lr={lr}, batch_size={batch_size} ===")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model_temp = SepsisMultitaskCascade(n_features=len(feature_cols), dim_z=dim_z).to(device)
    optimizer = torch.optim.Adam(model_temp.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

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

    if miglior_auroc_val_combo > miglior_auroc_globale:
        miglior_auroc_globale = miglior_auroc_val_combo
        migliori_iperparametri = (dim_z, lr, batch_size)
        migliori_pesi = pesi_migliori_combo

print("\n=== RISULTATO FINALE RICERCA IPERPARAMETRI ===")
print("Migliori iperparametri (dim_z, lr, batch_size):", migliori_iperparametri)
print("Miglior AUROC sepsi validation:", miglior_auroc_globale)

dim_z_finale, lr_finale, batch_size_finale = migliori_iperparametri
model = SepsisMultitaskCascade(n_features=len(feature_cols), dim_z=dim_z_finale).to(device)
model.load_state_dict(migliori_pesi)
model.eval()

with torch.no_grad():
    p_sepsis_val, p_inf_val, p_org_val = model(X_val_device)
    p_sepsis_val = p_sepsis_val.cpu().numpy()
    p_inf_val = p_inf_val.cpu().numpy()
    p_org_val = p_org_val.cpu().numpy()

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

# Valutazione finale sul TEST SET 
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

# calcolo le predizioni binarie (soglia 0.5), servono sia per le metriche sotto sia per l'utilità clinica
t_sepsis_val = (p_sepsis_val > 0.5).astype(int)
t_sepsis_test = (p_sepsis_test > 0.5).astype(int)

# metriche complete sul task sepsi (validation e test)
accuracy_val = accuracy_score(Y_val_sepsi, t_sepsis_val)
precision_val = precision_score(Y_val_sepsi, t_sepsis_val)
recall_val = recall_score(Y_val_sepsi, t_sepsis_val)
f1_val = f1_score(Y_val_sepsi, t_sepsis_val)
auprc_val = average_precision_score(Y_val_sepsi, p_sepsis_val)

accuracy_test = accuracy_score(Y_test_sepsi, t_sepsis_test)
precision_test = precision_score(Y_test_sepsi, t_sepsis_test)
recall_test = recall_score(Y_test_sepsi, t_sepsis_test)
f1_test = f1_score(Y_test_sepsi, t_sepsis_test)
auprc_test = average_precision_score(Y_test_sepsi, p_sepsis_test)

print("\n--- Risultati Validation Set (sepsi, metriche complete) ---")
print("AUROC:", auroc_sepsis, "\nAUPRC:", auprc_val, "\nAccuracy:", accuracy_val, "\nPrecision:", precision_val, "\nRecall:", recall_val, "\nF1 Score:", f1_val)

print("\n--- Risultati Test Set (valutazione finale) ---")
print("AUROC sepsi (test):", auroc_sepsis_test)
print("AUROC infezione (test):", auroc_inf_test)
print("AUROC organo (test):", auroc_org_test)
print("AUPRC sepsi (test):", auprc_test)
print("Accuracy sepsi (test):", accuracy_test)
print("Precision sepsi (test):", precision_test)
print("Recall sepsi (test):", recall_test)
print("F1 Score sepsi (test):", f1_test)

print("\n--- Risultati Train Set (confronto overfitting) ---")
print("AUROC sepsi (train):", auroc_sepsis_train, " vs validation:", auroc_sepsis)
print("AUROC infezione (train):", auroc_inf_train, " vs validation:", auroc_inf)
print("AUROC organo (train):", auroc_org_train, " vs validation:", auroc_org)

# Utilità clinica
punteggi_cascade_val = normalizza_punteggio(val_set["hours_to_sepsis"], val_set["is_sepsis"], t_sepsis_val.flatten())
print("\nUtilità clinica normalizzata Multitask Cascade (validation):", punteggi_cascade_val)

punteggi_cascade_test = normalizza_punteggio(test_set["hours_to_sepsis"], test_set["is_sepsis"], t_sepsis_test.flatten())
print("Utilità clinica normalizzata Multitask Cascade (test):", punteggi_cascade_test)