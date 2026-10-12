
import json
import subprocess
import yaml
import pandas as pd
import mlflow

# Connexion au serveur MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp3-dvc")

# Charger les hyperparamètres du modèle
P = yaml.safe_load(open("params.yaml", encoding="utf-8"))

# Récupérer l'empreinte MD5 des données suivies par DVC
ticket = yaml.safe_load(
    open("data/transactions.csv.dvc", encoding="utf-8")
)["outs"][0]

# Charger les performances du modèle
mets = json.load(open("reports/metrics.json"))

# Récupérer le commit Git actuel
commit = subprocess.getoutput("git rev-parse --short HEAD")

# Compter les transactions
n = len(pd.read_csv("data/transactions.csv"))

# Enregistrer une expérience MLflow
with mlflow.start_run(run_name=f"donnees-{ticket['md5'][:8]}"):

    mlflow.log_params(P["entrainement"])
    mlflow.log_metrics(mets)

    mlflow.set_tags({
        "commit": commit,
        "donnees_md5": ticket["md5"],
        "donnees_lignes": n
    })

    mlflow.log_artifact("dvc.lock")

print(
    "run MLflow cree : donnees",
    ticket["md5"][:8],
    "-",
    n,
    "lignes"
)
