from hi4lines_insp.main_base_oneoff import one_trial_train
from hydra import initialize, compose
import optuna
import psycopg2
import os

with initialize(config_path="../configs/"):
    cfg = compose(config_name="base")  # exp1.yaml with defaults key

# {'epochs': 155, 'lr': 0.03769210004835597, 'swa_start': 79, 'weight_decay': 2.266799648737876e-05, 'momentum': 0.9034986413496253, 'swa_lr': 0.001974199005673091}

def run_baseline(id, params):
    return one_trial_train(id, params['epochs'], params['lr'], params['weight_decay'], params['momentum'], cfg)

def main():
    study_name = 'coral_25_06_v2' #cfg.training.study_name
    storage = "postgresql+psycopg2://optuna_user:secretpass@localhost:5432/optuna_db"
    study = optuna.load_study(study_name=study_name, storage=storage)
    t = study.trials[0]
    print("Trial #{}: value={}, params={}".format(
        t.number, t.value, t.params
    ))
    
    print(t.params)
    run_baseline(0, t.params)
if __name__ == "__main__":
    main()
