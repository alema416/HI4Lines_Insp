from hi4lines_insp.main_base_oneoff import one_trial_train
from hydra import initialize, compose
import optuna
import psycopg2
import os

with initialize(config_path="../configs/"):
    cfg = compose(config_name="base")  # exp1.yaml with defaults key

def run_baseline(id, params):
    return one_trial_train(id, params['epochs'], params['lr'], params['weight_decay'], params['momentum'], cfg)

def main():
    study_name = cfg.training.study_name

    usrnm = cfg.secrets.usrnm
    pswrd = cfg.secrets.pswrd
    network = cfg.secrets.network
    port = cfg.secrets.port
    database_name = cfg.secrets.database_name

    storage = f"postgresql+psycopg2://{usrnm}:{pswrd}@{network}:{port}/{database_name}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    RUN_ID = cfg.training.bslnid
    t = study.trials[RUN_ID]
    print("Trial #{}: value={}, params={}".format(
        t.number, t.value, t.params
    ))
    run_baseline(RUN_ID, t.params)

if __name__ == "__main__":
    main()
