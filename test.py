import pandas as pd

# nomi dei file di input/output e parametri generali.
# CHUNKSIZE serve perché chartevents/labevents sono enormi (decine di milioni di righe):
# non entrano in RAM tutti insieme, quindi li leggo a blocchi da 2 milioni di righe alla volta
INPUT_SEPSIS_CSV = "sepsis3.csv"
INPUT_CONTROL_CSV = "control_cases.csv"
OUTPUT_CSV = "sepsis3_hourly_labeled.csv"
DATA_DIR = "Data"
CHUNKSIZE = 2_000_000

# MIMIC-IV registra la stessa identica misura con itemid diversi a seconda dello strumento
# o del reparto (es. la SBP ha più id possibili). Qui creo una mappa itemid -> nome variabile
# così riconduco tutti gli id a un unico nome pulito e non mi ritrovo 3 colonne per la stessa cosa
CHART_MAP = {
    220045: "HR",
    220052: "MAP",
    220277: "O2Sat",
    220210: "RR",
    226329: "Temp",
    220050: "SBP",
    225309: "SBP",
    228232: "SBP",
    220051: "DBP",
    225310: "DBP",
    228151: "DBP",
    224690: "Resp",
    224689: "Resp",
    224688: "Resp",
    228640: "EtCO2",
    223835: "FiO2",
    227010: "FiO2",
    220224: "PaCO2",
    227036: "PaCO2",
    220227: "SaO2",
    225165: "BaseExcess",
    220994: "HCO3",
    227443: "HCO3",
    226759: "HCO3",
}

# stessa logica di CHART_MAP ma per gli esami di laboratorio (labevents):
# tanti itemid che puntano alla stessa variabile, li unifico tutti sotto un nome solo
LAB_MAP = {
    50820: "PH",
    50831: "PH",
    50912: "Creatinine",
    52546: "Creatinine",
    50813: "Lactate",
    52442: "Lactate",
    50885: "Bilirubin_total",
    50884: "Bilirubin_total",
    53089: "Bilirubin_total",
    51301: "WBC",
    51755: "WBC",
    51756: "WBC",
    51265: "Platelets",
    53189: "Platelets",
    50878: "AST",
    51006: "BUN",
    52647: "BUN",
    50863: "Alkalinephos",
    53086: "Alkalinephos",
    50893: "Calcium",
    51066: "Calcium",
    50902: "Chloride",
    52535: "Chloride",
    50883: "Bilirubin_direct",
    50931: "Glucose",
    52569: "Glucose",
    50960: "Magnesium",
    50970: "Phosphate",
    50971: "Potassium",
    52610: "Potassium",
    51002: "TroponinI",
    52642: "TroponinI",
    50810: "Hct",
    51221: "Hct",
    50811: "Hgb",
    51222: "Hgb",
    51275: "PTT",
    51214: "Fibrinogen",
    52116: "Fibrinogen",
    50816: "FiO2",
    50802: "BaseExcess",
    50803: "HCO3",
    50818: "PaCO2",
    50817: "SaO2",
}

# lista di tutte le variabili numeriche (vitali + lab) che diventeranno le feature del modello.
# la tengo separata perché mi serve in più punti: conversione a numerico, forward fill e imputazione
NUMERIC_COLS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "PH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
]

# quando in una stessa ora ho più valori dello stesso esame devo decidere COME aggregarli.
# scelgo sempre il valore "peggiore" dal punto di vista clinico: max per quelli dove più alto = più grave
# (lattato, creatinina, bilirubina...), min per quelli dove più basso = più grave (piastrine, fibrinogeno, PH).
# tutto ciò che non è qui userà la media come default (vedi LAB_AGGREGATIONS.get(..., "mean") più sotto)
LAB_AGGREGATIONS = {
    "PH": "min",
    "Creatinine": "max",
    "Lactate": "max",
    "Bilirubin_total": "max",
    "Bilirubin_direct": "max",
    "WBC": "max",
    "Platelets": "min",
    "AST": "max",
    "BUN": "max",
    "Alkalinephos": "max",
    "TroponinI": "max",
    "Hct": "mean",
    "Hgb": "mean",
    "PTT": "max",
    "Fibrinogen": "min",
}


def carica_pazienti():
    # leggo i casi di sepsi e controllo subito che ci sia la colonna con l'onset:
    # senza quella non posso costruire la label, quindi meglio fermarsi qui con un errore chiaro
    sepsis_cases = pd.read_csv(INPUT_SEPSIS_CSV)
    if "sepsis_onset_time" not in sepsis_cases.columns:
        raise ValueError("Aggiungere la colonna 'sepsis_onset_time' al file sepsis3.csv.")

    # converto l'onset in datetime, così dopo posso fare i confronti temporali per la label
    sepsis_cases["sepsis_onset_time"] = pd.to_datetime(
        sepsis_cases["sepsis_onset_time"], errors="coerce"
    )
    # questi sono i settici "veri", li marco subito come positivi
    sepsis_cases["is_sepsis"] = True

    # leggo i controlli e converto in datetime le due colonne temporali che mi servono
    control_cases = pd.read_csv(INPUT_CONTROL_CSV)
    control_cases["suspected_infection_time"] = pd.to_datetime(
        control_cases["suspected_infection_time"], errors="coerce"
    )
    control_cases["sofa_time"] = pd.to_datetime(control_cases["sofa_time"], errors="coerce")
    # un controllo lo considero settico solo se ha SIA il sospetto di infezione SIA il SOFA:
    # se manca uno dei due non rispetta la definizione Sepsis-3
    control_cases["is_sepsis"] = (
        control_cases["suspected_infection_time"].notna()
        & control_cases["sofa_time"].notna()
    )
    # per i controlli costruisco l'onset come il più tardivo tra infezione e SOFA
    # (se ho entrambi i tempi), altrimenti lo lascio NaT
    control_cases["sepsis_onset_time"] = control_cases.apply(
        lambda row: max(row["suspected_infection_time"], row["sofa_time"])
        if pd.notna(row["suspected_infection_time"]) and pd.notna(row["sofa_time"])
        else pd.NaT,
        axis=1,
    )

    # tolgo dai controlli tutti gli stay che compaiono già tra i settici,
    # così non rischio di avere lo stesso ricovero in entrambi i gruppi
    sepsis_stays = set(sepsis_cases["stay_id"].unique())
    control_cases = control_cases[~control_cases["stay_id"].isin(sepsis_stays)].copy()

    # parto dalle colonne chiave dei controlli e poi mi riattacco tutte le colonne extra
    # (sofa_time, suspected_infection_time, ecc.) che mi serviranno più avanti per lo score clinico
    control_base = control_cases[
        ["subject_id", "hadm_id", "stay_id", "sepsis_onset_time", "is_sepsis"]
    ].copy()
    extra_control_columns = [
        column
        for column in control_cases.columns
        if column not in control_base.columns and column != "is_sepsis"
    ]
    for column in extra_control_columns:
        control_base[column] = control_cases[column].values

    # stessa cosa per i settici: chiavi davanti e tutto il resto delle colonne dietro
    sepsis_base = sepsis_cases[
        ["subject_id", "hadm_id", "stay_id", "sepsis_onset_time", "is_sepsis"]
        + [
            column
            for column in sepsis_cases.columns
            if column
            not in ["subject_id", "hadm_id", "stay_id", "sepsis_onset_time", "is_sepsis"]
        ]
    ].copy()

    # unisco settici + controlli in un'unica coorte
    pazienti = pd.concat([sepsis_base, control_base], ignore_index=True)
    # se per caso uno stay è duplicato, ordino mettendo i settici davanti e tengo solo la prima riga:
    # in pratica in caso di conflitto vince sempre l'etichetta "sepsi"
    pazienti = (
        pazienti.sort_values("is_sepsis", ascending=False)
        .drop_duplicates(subset=["subject_id", "hadm_id", "stay_id"])
        .reset_index(drop=True)
    )

    # un po' di stampe di controllo per vedere che i numeri abbiano senso
    print(f"Casi di controllo dopo la rimozione degli overlap: {len(control_cases)}")
    print(f"Stay totali nella coorte finale: {len(pazienti)}")
    print(f"  Stay con sepsi: {int(pazienti['is_sepsis'].sum())}")
    print(f"  Stay senza sepsi: {int((~pazienti['is_sepsis']).sum())}")

    return pazienti


def load_eligible_stays(pazienti):
    # mi servono ricoveri (admissions), permanenze in ICU (icustays) e anagrafica (patients)
    admissions = pd.read_csv(f"{DATA_DIR}/admissions.csv.gz", usecols=["subject_id", "hadm_id"])

    icu_stays = pd.read_csv(
        f"{DATA_DIR}/icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )
    # converto entrata/uscita dall'ICU in datetime e rinomino per non confonderle con altri tempi
    icu_stays["intime"] = pd.to_datetime(icu_stays["intime"], errors="coerce")
    icu_stays["outtime"] = pd.to_datetime(icu_stays["outtime"], errors="coerce")
    icu_stays = icu_stays.rename(columns={"intime": "icu_intime", "outtime": "icu_outtime"})

    patients = pd.read_csv(
        f"{DATA_DIR}/patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )

    # conto quanti ricoveri distinti ha ogni paziente: mi serve per tenere solo chi ne ha esattamente uno
    admission_counts = (
        admissions.groupby("subject_id")["hadm_id"].nunique().reset_index(name="n_hadm")
    )

    # parto dalle chiavi della coorte e ci attacco ICU, anagrafica e conteggio ricoveri
    pazienti_keys = pazienti[["subject_id", "hadm_id", "stay_id"]].drop_duplicates()
    demographics = (
        pazienti_keys
        .merge(icu_stays, on=["subject_id", "hadm_id", "stay_id"], how="inner")
        .merge(patients, on="subject_id", how="left")
        .merge(admission_counts, on="subject_id", how="left")
    )

    # in MIMIC l'età vera non c'è per privacy: la ricostruisco da anchor_age + (anno ICU - anchor_year)
    demographics["Age"] = demographics["anchor_age"] + (
        demographics["icu_intime"].dt.year - demographics["anchor_year"]
    )
    demographics["Gender"] = demographics["gender"]
    # quante ore è durata la permanenza in ICU (uscita - entrata, in ore)
    demographics["icu_hours"] = (
        demographics["icu_outtime"] - demographics["icu_intime"]
    ).dt.total_seconds() / 3600

    # criteri di inclusione: solo adulti (>=18), almeno 24h in ICU (servono abbastanza ore di dati)
    # e un solo ricovero, così evito pazienti con storie cliniche multiple che complicano l'analisi
    eligible_stays = demographics[
        (demographics["Age"] >= 18)
        & (demographics["icu_hours"] >= 24)
        & (demographics["n_hadm"] == 1)
    ].copy()
    # n_hours = numero intero di ore della griglia oraria che costruirò per ogni stay
    eligible_stays["n_hours"] = eligible_stays["icu_hours"].astype(int)

    print(f"Stay eleggibili dopo i filtri clinici e temporali: {len(eligible_stays)}")
    # se non resta nessuno qualcosa è andato storto, meglio fermarsi
    if eligible_stays.empty:
        print("Nessun stay soddisfa i criteri richiesti.")
        raise SystemExit

    return eligible_stays


def build_hourly_grid(eligible_stays):
    # qui costruisco lo "scheletro" temporale: per ogni stay creo una riga per ogni ora di permanenza.
    # è la griglia vuota su cui poi andrò ad appoggiare i vitali e i lab ora per ora
    hourly_rows = []
    for _, stay in eligible_stays.iterrows():
        for hour_index in range(int(stay["n_hours"])):
            hourly_rows.append(
                {
                    "subject_id": stay["subject_id"],
                    "hadm_id": stay["hadm_id"],
                    "stay_id": stay["stay_id"],
                    "Gender": stay["Gender"],
                    "Age": stay["Age"],
                    "icu_hours": stay["icu_hours"],
                    "hour_index": hour_index,
                    # inizio e fine di questa finestra di un'ora, partendo dall'ingresso in ICU
                    "hour_start": stay["icu_intime"] + pd.Timedelta(hours=hour_index),
                    "hour_end": stay["icu_intime"] + pd.Timedelta(hours=hour_index + 1),
                }
            )

    hourly_grid = pd.DataFrame(hourly_rows)
    print(f"Righe create nella griglia oraria: {len(hourly_grid)}")
    return hourly_grid


def collect_matching_chunks(file_path, usecols, item_ids, join_keys):
    # funzione generica per leggere a blocchi i file giganti (chartevents/labevents).
    # per ogni blocco tengo solo gli itemid che mi interessano e solo i pazienti della mia coorte,
    # così butto via subito tutto il resto e non saturo la memoria
    matching_chunks = []
    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=CHUNKSIZE):
        # primo filtro: solo le righe con un itemid presente nella mia mappa
        filtered_chunk = chunk[chunk["itemid"].isin(item_ids)]
        if filtered_chunk.empty:
            continue

        # secondo filtro: tengo solo i pazienti/stay eleggibili (join sulle chiavi)
        filtered_chunk = filtered_chunk.merge(join_keys, on=list(join_keys.columns), how="inner")
        if filtered_chunk.empty:
            continue

        matching_chunks.append(filtered_chunk)

    return matching_chunks


def aggregate_chart_events(eligible_stays, hourly_grid):
    # i vitali (chartevents) sono già legati allo stay_id, quindi uso quello come chiave
    eligible_keys = eligible_stays[["subject_id", "hadm_id", "stay_id"]].drop_duplicates()
    chart_chunks = collect_matching_chunks(
        file_path=f"{DATA_DIR}/chartevents.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        item_ids=set(CHART_MAP),
        join_keys=eligible_keys,
    )

    # se non trovo nulla restituisco solo le chiavi vuote, così il merge a valle non si rompe
    if not chart_chunks:
        return hourly_grid[["subject_id", "hadm_id", "stay_id", "hour_index"]].copy()

    # rimetto insieme tutti i blocchi filtrati in un unico dataframe
    chart_events = pd.concat(chart_chunks, ignore_index=True)
    chart_events["charttime"] = pd.to_datetime(chart_events["charttime"], errors="coerce")
    # traduco l'itemid nel nome variabile leggibile usando la mappa
    chart_events["variable"] = chart_events["itemid"].map(CHART_MAP)
    # butto via le righe senza orario, senza valore o senza variabile riconosciuta
    chart_events = chart_events.dropna(subset=["charttime", "valuenum", "variable"])

    # mi attacco l'orario di ingresso in ICU per capire a quale ora appartiene ogni misura
    chart_events = chart_events.merge(
        eligible_stays[["subject_id", "hadm_id", "stay_id", "icu_intime", "n_hours"]],
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
    )
    # calcolo a quante ore dall'ingresso è stata presa la misura
    chart_events["hours_from_icu"] = (
        chart_events["charttime"] - chart_events["icu_intime"]
    ).dt.total_seconds() / 3600.0
    # tengo solo le misure dentro la finestra di permanenza (da 0 fino a n_hours):
    # scarto roba presa prima dell'ingresso o dopo l'uscita
    chart_events = chart_events[
        (chart_events["hours_from_icu"] >= 0)
        & (chart_events["hours_from_icu"] < chart_events["n_hours"])
    ].copy()
    # tronco all'intero per assegnare ogni misura alla sua ora (es. 3.7h -> ora 3)
    chart_events["hour_index"] = chart_events["hours_from_icu"].astype(int)

    # raggruppo per stay + ora + variabile e faccio la media dei valori dentro la stessa ora,
    # poi con unstack "apro" le variabili in colonne (da formato lungo a formato largo)
    chart_hourly = (
        chart_events.groupby(
            ["subject_id", "hadm_id", "stay_id", "hour_index", "variable"]
        )["valuenum"]
        .mean()
        .unstack()
        .reset_index()
    )

    print(f"Shape delle variabili da chartevents: {chart_hourly.shape}")
    return chart_hourly


def aggregate_lab_events(eligible_stays, hourly_grid):
    # i lab (labevents) NON hanno lo stay_id, sono legati solo al ricovero (hadm_id),
    # quindi qui la chiave del join è subject_id + hadm_id e non lo stay
    eligible_admissions = eligible_stays[["subject_id", "hadm_id"]].drop_duplicates()
    lab_chunks = collect_matching_chunks(
        file_path=f"{DATA_DIR}/labevents.csv.gz",
        usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        item_ids=set(LAB_MAP),
        join_keys=eligible_admissions,
    )

    # parto dalle chiavi della griglia oraria su cui andrò ad appoggiare i lab
    base_hourly = (
        hourly_grid[["subject_id", "hadm_id", "stay_id", "hour_index"]]
        .drop_duplicates()
        .copy()
    )
    # se non ho trovato lab restituisco solo le chiavi, senza colonne aggiuntive
    if not lab_chunks:
        return base_hourly

    lab_events = pd.concat(lab_chunks, ignore_index=True)
    lab_events["charttime"] = pd.to_datetime(lab_events["charttime"], errors="coerce")
    lab_events["variable"] = lab_events["itemid"].map(LAB_MAP)
    lab_events = lab_events.dropna(subset=["charttime", "valuenum", "variable"])

    # collego il tempo dell'esame all'ingresso in ICU. siccome i lab sono sul ricovero,
    # il merge è solo su subject_id + hadm_id (lo stay viene "ereditato" da eligible_stays)
    lab_events = lab_events.merge(
        eligible_stays[["subject_id", "hadm_id", "stay_id", "icu_intime", "n_hours"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    lab_events["hours_from_icu"] = (
        lab_events["charttime"] - lab_events["icu_intime"]
    ).dt.total_seconds() / 3600.0
    # come prima: tengo solo gli esami dentro la finestra di permanenza
    lab_events = lab_events[
        (lab_events["hours_from_icu"] >= 0)
        & (lab_events["hours_from_icu"] < lab_events["n_hours"])
    ].copy()
    lab_events["hour_index"] = lab_events["hours_from_icu"].astype(int)

    # qui NON faccio sempre la media: per ogni esame uso l'aggregazione clinica decisa in LAB_AGGREGATIONS
    # (max/min/mean), perché per certi esami conta il valore peggiore e non quello medio.
    # ogni variabile la attacco come nuova colonna con un merge a sinistra sulla griglia
    for variable_name, group in lab_events.groupby("variable"):
        aggregation = LAB_AGGREGATIONS.get(variable_name, "mean")
        aggregated = (
            group.groupby(["subject_id", "hadm_id", "stay_id", "hour_index"])["valuenum"]
            .agg(aggregation)
            .reset_index(name=variable_name)
        )
        base_hourly = base_hourly.merge(
            aggregated,
            on=["subject_id", "hadm_id", "stay_id", "hour_index"],
            how="left",
        )

    print(f"Shape delle variabili da labevents: {base_hourly.shape}")
    return base_hourly


def assemble_final_table(hourly_grid, chart_hourly, lab_hourly, pazienti):
    # parto dalla griglia oraria e ci attacco prima i vitali e poi i lab, ora per ora.
    # uso left join così mantengo TUTTE le ore anche dove non ho misure (resteranno NaN per ora)
    final_table = hourly_grid.merge(
        chart_hourly, on=["subject_id", "hadm_id", "stay_id", "hour_index"], how="left"
    )
    final_table = final_table.merge(
        lab_hourly, on=["subject_id", "hadm_id", "stay_id", "hour_index"], how="left"
    )

    # riattacco le info della coorte (onset, is_sepsis, tempi clinici...) a livello di stay
    pazienti_info = pazienti.drop_duplicates(subset=["subject_id", "hadm_id", "stay_id"]).copy()
    final_table = final_table.merge(
        pazienti_info, on=["subject_id", "hadm_id", "stay_id"], how="left"
    )

    # mi assicuro che tutte le colonne numeriche esistano (se una variabile non è mai comparsa la creo vuota)
    # e le forzo a numerico arrotondando a 2 decimali, così il CSV resta pulito
    for column in NUMERIC_COLS:
        if column not in final_table.columns:
            final_table[column] = pd.NA
        final_table[column] = pd.to_numeric(final_table[column], errors="coerce").round(2)

    # l'età la arrotondo all'intero
    final_table["Age"] = pd.to_numeric(final_table["Age"], errors="coerce").round(0)
    return final_table


def add_sepsis_label(final_table):
    # mi assicuro che onset e fine ora siano datetime per poterli confrontare
    final_table["sepsis_onset_time"] = pd.to_datetime(
        final_table["sepsis_onset_time"], errors="coerce"
    )
    final_table["hour_end"] = pd.to_datetime(final_table["hour_end"])

    # qui creo la label come nel PhysioNet Challenge 2019: l'ora è positiva (1) a partire da 6 ore
    # PRIMA dell'onset e resta positiva per tutte le ore successive.
    # in pratica chiedo al modello di "accendersi" nelle 6 ore che precedono la sepsi
    final_table["label_sepsis_6h"] = (
        final_table["hour_end"]
        >= (final_table["sepsis_onset_time"] - pd.Timedelta(hours=6))
    ).astype(int)

    # qualche statistica per controllare il bilanciamento della label (di solito è molto sbilanciata)
    positive_rows = final_table["label_sepsis_6h"].sum()
    total_rows = len(final_table)
    positive_stays = final_table.groupby("stay_id")["label_sepsis_6h"].max().sum()
    negative_stays = final_table.groupby("stay_id")["label_sepsis_6h"].max().eq(0).sum()

    print("\n=== Label sepsis_6h (PhysioNet Challenge 2019) ===")
    print(f"Righe positive: {positive_rows} / {total_rows} ({100 * positive_rows / total_rows:.2f}%)")
    print(f"Stay con almeno un'ora positiva: {int(positive_stays)}")
    print(f"Stay sempre negativi: {int(negative_stays)}")

    # controllo veloce solo sui settici: mi aspetto che abbiano sia ore a 0 (prima) sia ore a 1 (vicino/dopo onset)
    septic_rows = final_table[final_table["is_sepsis"] == True].copy()
    print("\nControllo rapido sui soli pazienti settici:")
    print(f"Ore con label 0: {len(septic_rows[septic_rows['label_sepsis_6h'] == 0])}")
    print(f"Ore con label 1: {len(septic_rows[septic_rows['label_sepsis_6h'] == 1])}")

    return final_table


def reorder_columns(final_table, pazienti):
    # rimetto le colonne in un ordine sensato: prima gli identificativi e le info di base,
    # poi le colonne extra della coorte, poi tutte le feature numeriche e infine la label
    ordered_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "Gender",
        "Age",
        "icu_hours",
        "is_sepsis",
        "hour_index",
        "hour_start",
        "hour_end",
    ]

    # aggiungo le colonne della coorte non ancora presenti, ma salto sepsis_onset_time
    # perché serviva solo a costruire la label e non voglio tenerla nel file finale
    for column in pazienti.columns:
        if column not in ordered_columns and column != "sepsis_onset_time":
            ordered_columns.append(column)

    # poi tutte le feature numeriche
    for column in NUMERIC_COLS:
        if column not in ordered_columns:
            ordered_columns.append(column)

    # e per ultima la label
    if "label_sepsis_6h" not in ordered_columns:
        ordered_columns.append("label_sepsis_6h")

    # tengo solo le colonne che esistono davvero ed elimino l'onset dal dataframe finale
    ordered_columns = [column for column in ordered_columns if column in final_table.columns]
    final_table = final_table.drop(columns=["sepsis_onset_time"], errors="ignore")
    return final_table[ordered_columns]


def fill_missing_values(final_table):
    # ordino per paziente e per ora: fondamentale perché il forward fill deve seguire la linea temporale
    final_table = final_table.sort_values(
        by=["subject_id", "hadm_id", "stay_id", "hour_index"]
    ).copy()

    print("\nValori mancanti prima del forward fill:")
    print(final_table[NUMERIC_COLS].isna().sum())

    # forward fill DENTRO ogni stay: se in un'ora non ho una misura, riporto l'ultima misurata.
    # ha senso clinicamente (un valore vale finché non ne arriva uno nuovo) e il groupby evita
    # che un valore "sconfini" da un paziente all'altro
    final_table[NUMERIC_COLS] = (
        final_table.groupby(["subject_id", "hadm_id", "stay_id"])[NUMERIC_COLS].ffill()
    )

    print("\nValori mancanti dopo il forward fill:")
    print(final_table[NUMERIC_COLS].isna().sum())

    # quello che resta vuoto (es. inizio stay senza nessuna misura precedente) lo metto a -1:
    # è un valore "sentinella" che dice al modello "qui non ho dato", senza confonderlo con uno zero vero
    final_table[NUMERIC_COLS] = final_table[NUMERIC_COLS].fillna(-1)

    print("\nNumero di valori imputati a -1 per feature:")
    print((final_table[NUMERIC_COLS] == -1).sum())

    # stessa logica per l'età eventualmente mancante
    if "Age" in final_table.columns:
        final_table["Age"] = final_table["Age"].fillna(-1)

    return final_table


def main():
    # pipeline completa in ordine: costruisco la coorte, filtro gli stay validi, creo la griglia oraria,
    # aggrego vitali e lab, assemblo tutto, metto la label, riordino le colonne e imputo i mancanti
    pazienti = carica_pazienti()
    eligible_stays = load_eligible_stays(pazienti)
    hourly_grid = build_hourly_grid(eligible_stays)
    chart_hourly = aggregate_chart_events(eligible_stays, hourly_grid)
    lab_hourly = aggregate_lab_events(eligible_stays, hourly_grid)

    final_table = assemble_final_table(hourly_grid, chart_hourly, lab_hourly, pazienti)
    final_table = add_sepsis_label(final_table)
    final_table = reorder_columns(final_table, pazienti)
    final_table = fill_missing_values(final_table)

    # ultimo sguardo prima di salvare: anteprima, shape e nomi delle colonne
    print(final_table.head())
    print(f"\nShape finale: {final_table.shape}")
    print(f"Colonne finali: {list(final_table.columns)}")

    # salvo il dataset orario etichettato, che diventerà l'input del file di training
    final_table.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV salvato in: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()