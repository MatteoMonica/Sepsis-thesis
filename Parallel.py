import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import random
import shap
import matplotlib.pyplot as plt

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

def punteggio_ottimale(hours_to_sepsis, is_sepsis):
    return max(
        calcola_punteggio(hours_to_sepsis, 1, is_sepsis),
        calcola_punteggio(hours_to_sepsis, 0, is_sepsis)
    )            

def normalizza_punteggio(hours_to_sepsis_list,is_sepsis_list,prediction):
   U_totale = sum([calcola_punteggio(ore, pred, sepsi) for ore, pred, sepsi in zip(hours_to_sepsis_list, prediction, is_sepsis_list)])
   U_no_predictions=sum([calcola_punteggio(ore, 0, sepsi) for ore, sepsi in zip(hours_to_sepsis_list, is_sepsis_list)])
   U_optimal=sum([punteggio_ottimale(ore,sepsi) for ore,sepsi in zip(hours_to_sepsis_list, is_sepsis_list)])
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

# Stesse colonne da escludere ( incluse quelle anti-leakage) 
colonne_da_escludere = ["subject_id","hadm_id","stay_id","label_sepsis_6h","label_infection_6h","label_organ_6h","Gender",
                        "hour_start","hour_end","intime","antibiotic_time","culture_time","suspected_infection_time","sofa_time",
                        "sepsis3","sepsis_onset","is_sepsis","sofa_score","sepsis_onset_hour","hours_to_sepsis","FiO2","HCO3","PaCO2",
                        "TroponinI","EtCO2","SaO2","anchor_year_group","anchor_age","anchor_year","hour_index","respiration","coagulation",
                        "liver","cardiovascular","cns","renal","icu_hours"]

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
print("Shape Y_train_sepsi_tensor:", Y_train_sepsi_tensor.shape)

class SepsisMultitaskParallelo(nn.Module):
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
        self.sepsis_head = nn.Linear(dim_z, 1)

    def forward(self, x):
        z = self.encoder(x)
        logit_inf = self.infezione(z)
        logit_org = self.organo(z)
        logit_sepsis = self.sepsis_head(z)

        p_inf = torch.sigmoid(logit_inf)
        p_org = torch.sigmoid(logit_org)
        p_sepsis = torch.sigmoid(logit_sepsis)

        return p_sepsis, p_inf, p_org

combinazioni_multitask = [(dim_z, lr, batch_size)
                           for dim_z in [32, 64, 128]
                           for lr in [0.001, 0.0003, 0.0001]
                           for batch_size in [128, 256, 512]]

train_dataset = TensorDataset(X_train_tensor, Y_train_sepsi_tensor, Y_train_inf_tensor, Y_train_org_tensor)

X_val_device = X_val_tensor.to(device)

seeds = [42, 123, 7, 2024, 99]
# Traccio le tre AUROC (sepsi + i due task ausiliari) perché è proprio il punto del modello multitask:
# capire se imparare infezione/organo aiuta il task principale (sepsi). Le altre metriche
# (Accuracy, Precision, ecc.) restano solo su sepsi, per restare confrontabili con MLP/XGBoost/LSTM
risultati_multi_run = {
    "AUROC": [], "AUROC_infezione": [], "AUROC_organo": [],
    "AUPRC": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [], "Utilita": []
}
for seed_run in seeds:
    print(f"\n\n--------- SEED {seed_run} ---------")
    torch.manual_seed(seed_run)
    rng = random.Random(seed_run)
    combinazioni_scelte_multitask = rng.sample(combinazioni_multitask, 8)

    miglior_auroc_globale = 0.0
    migliori_iperparametri = None
    migliori_pesi = None

    for dim_z, lr, batch_size in combinazioni_scelte_multitask:
        print(f"\n=== Provo dim_z={dim_z}, lr={lr}, batch_size={batch_size} ===")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model_temp = SepsisMultitaskParallelo(n_features=len(feature_cols), dim_z=dim_z).to(device)
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

    print(f"\nMigliori iperparametri per seed {seed_run} (dim_z, lr, batch_size):", migliori_iperparametri)
    print("Miglior AUROC sepsi validation:", miglior_auroc_globale)

    dim_z_finale, lr_finale, batch_size_finale = migliori_iperparametri
    model = SepsisMultitaskParallelo(n_features=len(feature_cols), dim_z=dim_z_finale).to(device)
    model.load_state_dict(migliori_pesi)
    model.eval()
    # NB: qui prima calcolavo anche le metriche sul validation set e il confronto train vs validation
    # (per controllare overfitting). Le ho tolte perché con 5 seed diventerebbero 5 blocchi
    # di stampa ridondanti 
    # Valutazione finale sul TEST SET
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        p_sepsis_test, p_inf_test, p_org_test = model(X_test_device)

        p_sepsis_test = p_sepsis_test.cpu().numpy()
        p_inf_test = p_inf_test.cpu().numpy()
        p_org_test = p_org_test.cpu().numpy()

    auroc_sepsis_test = roc_auc_score(Y_test_sepsi, p_sepsis_test)
    auroc_inf_test = roc_auc_score(Y_test_inf, p_inf_test)
    auroc_org_test = roc_auc_score(Y_test_org, p_org_test)
    auprc_test = average_precision_score(Y_test_sepsi, p_sepsis_test)

    t_sepsis_test = (p_sepsis_test > 0.5).astype(int)
    accuracy_test = accuracy_score(Y_test_sepsi, t_sepsis_test)
    precision_test = precision_score(Y_test_sepsi, t_sepsis_test)
    recall_test = recall_score(Y_test_sepsi, t_sepsis_test)
    f1_test = f1_score(Y_test_sepsi, t_sepsis_test)

    print(f"\n--- Risultati Test Set (seed {seed_run}) ---")
    print("AUROC sepsi:", auroc_sepsis_test)
    print("AUROC infezione:", auroc_inf_test)
    print("AUROC organo:", auroc_org_test)
    print("AUPRC sepsi:", auprc_test)
    print("Accuracy sepsi:", accuracy_test)
    print("Precision sepsi:", precision_test)
    print("Recall sepsi:", recall_test)
    print("F1 Score sepsi:", f1_test)

    punteggi_parallelo_test = normalizza_punteggio(test_set["hours_to_sepsis"], test_set["is_sepsis"], t_sepsis_test.flatten())
    print("Media utilita' clinica Parallelo Test set:", punteggi_parallelo_test)

    risultati_multi_run["AUROC"].append(auroc_sepsis_test)
    risultati_multi_run["AUROC_infezione"].append(auroc_inf_test)
    risultati_multi_run["AUROC_organo"].append(auroc_org_test)
    risultati_multi_run["AUPRC"].append(auprc_test)
    risultati_multi_run["Accuracy"].append(accuracy_test)
    risultati_multi_run["Precision"].append(precision_test)
    risultati_multi_run["Recall"].append(recall_test)
    risultati_multi_run["F1 Score"].append(f1_test)
    risultati_multi_run["Utilita"].append(punteggi_parallelo_test)

    # Explainable AI SHAP per il modello multitask Parallelo
    model_cpu = model.to("cpu")
    model_cpu.eval()

    class WrapperOutput(nn.Module):
        def __init__(self, modello, indice_output):
            super().__init__()
            self.modello = modello
            self.indice_output = indice_output  # 0=sepsis, 1=inf, 2=org

        def forward(self, x):
            outputs = self.modello(x)  # tupla (p_sepsis, p_inf, p_org)
            return outputs[self.indice_output]

    # background campionato casualmente random_state fisso a 42 voglio lo stesso
    # background per ogni seed, così le differenze tra i plot dipendono solo dal modello,
    # non da quali pazienti ho pescato dal training set
    idx_bg = np.random.RandomState(42).choice(X_train_tensor.shape[0], size=100, replace=False)
    background = X_train_tensor[idx_bg].to("cpu")

    rng_shap = np.random.RandomState(42)
    test_sample_idx = rng_shap.choice(len(X_test_tensor), size=500, replace=False)
    test_sample = X_test_tensor[test_sample_idx].to("cpu")

    nomi_task = ["sepsis", "infezione", "organo"]

    for indice_output, nome_task in enumerate(nomi_task):
        print(f"\nCalcolo SHAP per il task: {nome_task} (seed {seed_run})")

        wrapper = WrapperOutput(model_cpu, indice_output)
        explainer = shap.GradientExplainer(wrapper, background)
        shap_values = explainer.shap_values(test_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values).reshape(len(test_sample), len(feature_cols))

        importanza_media = np.abs(shap_values).mean(axis=0)
        ordine = np.argsort(importanza_media)[::-1]
        feature_ordinate = [feature_cols[i] for i in ordine]
        valori_ordinati = importanza_media[ordine]

        plt.figure(figsize=(8, 10))
        plt.barh(feature_ordinate[:20][::-1], valori_ordinati[:20][::-1])
        plt.xlabel("Importanza media |valore SHAP|")
        plt.title(f"Multitask Parallelo - Importanza feature per task: {nome_task} (seed {seed_run})")
        plt.tight_layout()
        plt.savefig(f"Parallelo_{nome_task}_{seed_run}.png", dpi=150)  # seed nel nome, altrimenti si sovrascrivono ad ogni iterazione
        plt.close()

    print(f"\nGrafici SHAP multitask Parallelo salvati per tutti e 3 i task (seed {seed_run}).")

print(f"\n\n========== RISULTATI FINALI PARALLELO: MEDIA +/- DEV. STANDARD SU {len(seeds)} SEED ==========")
for chiave, valori in risultati_multi_run.items():
    media = np.mean(valori)
    std = np.std(valori)
    print(f"{chiave}: {media:.4f} +/- {std:.4f}")