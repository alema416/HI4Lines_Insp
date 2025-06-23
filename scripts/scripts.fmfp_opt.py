from hi4lines_insp.fmfp_pipeline import one_trial_train
from hydra import initialize, compose
import optuna
import psycopg2
import os

with initialize(config_path="../configs/"):
    cfg = compose(config_name="fmfp")  # exp1.yaml with defaults key

def objective(trial):
    
    epochs = trial.suggest_int("epochs", cfg.training.epochs_low, cfg.training.epochs_high)
    base_lr = trial.suggest_loguniform('lr', cfg.training.base_lr_low, cfg.training.base_lr_high)
    swa_start = trial.suggest_int("swa_start", int(epochs/2), int((3/4)*epochs))
    custom_weight_decay = trial.suggest_loguniform('weight_decay', cfg.training.weight_decay_low, cfg.training.weight_decay_high) 
    custom_momentum = trial.suggest_uniform('momentum', cfg.training.momentum_low, cfg.training.momentum_high) 
    swa_lr = trial.suggest_loguniform('swa_lr', cfg.fmfp.swa_lr_low, cfg.fmfp.swa_lr_high) 

    return one_trial_train(trial.number, epochs, base_lr, custom_weight_decay, custom_momentum, swa_start, swa_lr, cfg)

def main():
    study_name = 'postgress' #cfg.training.study_name
    #storage = f"sqlite:///{os.path.join(os.getcwd(), f'{study_name}.sqlite')}"
    storage = "postgresql+psycopg2://optuna_user:secretpass@optuna-postgres:5432/optuna_db"
    study = optuna.create_study(direction='minimize', load_if_exists=True, study_name = study_name, storage=storage)
    
    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(objective, n_trials=2, n_jobs=1)

    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)

if __name__ == "__main__":
    main()
