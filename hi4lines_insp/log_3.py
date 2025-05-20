#!/usr/bin/env python3
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from mlflow.tracking import MlflowClient

# 1) Point at your local mlruns/
here      = os.path.dirname(__file__)
mlruns    = os.path.join(here, "mlruns")
client    = MlflowClient(tracking_uri=f"file://{os.path.abspath(mlruns)}")

# 2) Experiment and the eight parent names
EXP_ID    = "994808004837205094"
PARENTS   = {
    "smiling-roo-885",
    "wise-robin-426",
    "welcoming-bat-556",
    "sincere-pug-237",
    "welcoming-flea-457",
    "abrasive-smelt-174",
    "monumental-lynx-469",
    "spiffy-bat-195",
}

# 3) Fetch every run in that experiment
all_runs = client.search_runs([EXP_ID], filter_string="")

# 4) Map parent_run_id → runName for just our eight roots
parent_ids = {
    r.info.run_id: r.data.tags.get("mlflow.runName", "")
    for r in all_runs
    if not r.data.tags.get("mlflow.parentRunId")
    and r.data.tags.get("mlflow.runName") in PARENTS
}

# 5) Collect all child runs under those parents
rows = []
df_diff = pd.DataFrame(columns=['run_name', 'diff_of_last_epochs_averages', 'val_hw_augrc', 'val_loss'])
for run in all_runs:
    pid = run.data.tags.get("mlflow.parentRunId")
    if pid in parent_ids:
        rows.append({
            "started":        datetime.fromtimestamp(run.info.start_time/1000),
            "run_name":       run.data.tags.get("mlflow.runName", run.info.run_id),
            "run_id":         run.info.run_id,
            # hyper‐params
            "lr":             run.data.params.get("lr"),
            "weight_decay":   run.data.params.get("weight_decay"),
            "momentum":       run.data.params.get("momentum"),
            # original metrics
            "train_acc":      run.data.metrics.get("train_acc"),
            "val_acc":        run.data.metrics.get("val_acc"),
            "test_acc":       run.data.metrics.get("test_acc"),
            # hardware metrics
            "acc_hw_train":   run.data.metrics.get("acc_hw_train"),
            "acc_hw_val":     run.data.metrics.get("acc_hw_val"),
            "acc_hw_test":    run.data.metrics.get("acc_hw_test"),
            "augrc_hw_train": run.data.metrics.get("augrc_hw_train"),
            "augrc_hw_val":   run.data.metrics.get("augrc_hw_val"),
            "augrc_hw_test":  run.data.metrics.get("augrc_hw_test"),
        })
        #print(f"\nRun ID: {run.info.run_id}")
        metrics_train_loss = client.get_metric_history(run.info.run_id, "train_loss")
        metrics_val_loss = client.get_metric_history(run.info.run_id, "val_loss")
        
        metrics_train_acc = client.get_metric_history(run.info.run_id, "train_acc")
        metrics_val_acc = client.get_metric_history(run.info.run_id, "val_acc")

        #print(f'{run.info.run_name}: swa_start: {run.data.params.get("swa_start")}')

        #print(run.info)
        if len(metrics_train_loss) > 1 and len(metrics_val_loss):
            train_loss_dict = {m.step: m.value for m in metrics_train_loss}
            val_loss_dict   = {m.step: m.value for m in metrics_val_loss}

            # Shared steps
            common_steps = sorted(set(train_loss_dict) & set(val_loss_dict))
            diff_loss = [abs(train_loss_dict[step] - val_loss_dict[step]) for step in common_steps]

            train_steps = sorted(train_loss_dict)
            last_10_train_steps = train_steps[-10:]
            
            last_10_train_values = [train_loss_dict[step] for step in last_10_train_steps]
            train_values = [train_loss_dict[step] for step in train_steps]
            avg_last_10_train = np.mean(last_10_train_values)
            val_steps = sorted(val_loss_dict)
            val_values = [val_loss_dict[step] for step in val_steps]
            val_loss_in_last_10 = [
                val_loss_dict[step] for step in last_10_train_steps if step in val_loss_dict
            ]
            avg_val_on_last_10_train_steps = np.mean(val_loss_in_last_10)
            #print(f'abs(avg_last_10_train - avg_val_on_last_10_train_steps)')
            plt.figure(figsize=(10, 5))
            plt.plot(train_steps, train_values, label="train_loss")
            plt.plot(val_steps, val_values, label="val_loss")
            plt.plot(common_steps, diff_loss, label="train_loss - val_loss", linestyle='--')
            if run.data.params.get("swa_start"):
                plt.axvline(x=int(run.data.params.get("swa_start")), color='green', linestyle='--', label='swa_start')
            plt.title(f"Loss Curve - {run.data.tags.get('mlflow.runName', run.info.run_id)}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            '''
            train_steps = [m.step for m in metrics_train_loss]
            train_values = [m.value for m in metrics_train_loss]

            val_steps = [m.step for m in metrics_val_loss]
            val_values = [m.value for m in metrics_val_loss]

            plt.figure(figsize=(10, 5))
            plt.plot(train_steps, train_values, label="train_loss")
            plt.plot(val_steps, val_values, label="val_loss")
            if run.data.params.get("swa_start"):
                plt.axvline(x=int(run.data.params.get("swa_start")), color='green', linestyle='--', label='swa_start')
            plt.title(f"Loss Curve - {run.data.tags.get('mlflow.runName', run.info.run_id)}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            '''
            plot_path = os.path.join('', f"{run.info.run_name}_loss_curve.png")
            plt.savefig(plot_path)
            plt.close()
            '''
            train_steps = [m.step for m in metrics_train_acc]
            train_values = [m.value for m in metrics_train_acc]

            val_steps = [m.step for m in metrics_val_acc]
            val_values = [m.value for m in metrics_val_acc]

            plt.figure(figsize=(10, 5))
            plt.plot(train_steps, train_values, label="train_acc")
            plt.plot(val_steps, val_values, label="val_acc")
            if run.data.params.get("swa_start"):
                plt.axvline(x=int(run.data.params.get("swa_start")), color='green', linestyle='--', label='swa_start')
            plt.title(f"Acc Curve - {run.data.tags.get('mlflow.runName', run.info.run_id)}")
            plt.xlabel("Epoch")
            plt.ylabel("Acc")
            plt.legend()
            plt.grid(True)

            plot_path = os.path.join('', f"{run.info.run_name}_acc_curve.png")
            plt.savefig(plot_path)
            plt.close()

            #pass
            #print(f'{len(metrics_train_loss)}, {len(metrics_val_loss)}')
            #print(f"Run {run.info.run_name} has per-epoch 'val_acc' metrics:")
            #for m in metrics:
            #    print(f"  Step {m.step}: val_acc = {m.value}")
            '''
            #print(f'train {np.mean(train_values[-30:])}')
            #print(f'val {np.mean(val_values[-30:])}')
            #print(f'diff {np.mean(diff_loss[-30:])}')
            df_diff.loc[len(df_diff)] = [run.info.run_name, abs(avg_last_10_train - avg_val_on_last_10_train_steps), run.data.metrics.get("augrc_hw_val"), metrics_val_loss[-1].value]
        else:
            print(f"Run {run.info.run_name} does NOT have per-epoch 'val_acc' metrics.")
df_diff.to_csv('./diffs.csv', index=False)
df_diff = df_diff.sort_values(by='diff_of_last_epochs_averages')
print(df_diff)
# 6) Build DataFrame and sort by start time
df = pd.DataFrame(rows)
df = df.sort_values("started")

# … after you build & sort df …

df = df.dropna(subset=[
    "val_acc","test_acc",
    "acc_hw_train","acc_hw_val","acc_hw_test",
    "augrc_hw_train","augrc_hw_val","augrc_hw_test"
], how="any")
df = df.round(2)
# now print—no rows containing NaN will remain
print(df[[
    "run_name",
    "acc_hw_train","acc_hw_val","acc_hw_test",
    "augrc_hw_train","augrc_hw_val","augrc_hw_test"
]].to_markdown(
    index=False,
    headers=[
      "run_name",
      "acc_hw_train","acc_hw_val","acc_hw_test",
      "augrc_hw_train","augrc_hw_val","augrc_hw_test"
    ]
))

# … your existing filtering + rounding …
subset = [
    "run_name",
    "acc_hw_train","acc_hw_val","acc_hw_test",
    "augrc_hw_train","augrc_hw_val","augrc_hw_test"
]
out_df = df[subset]

# save to CSV
csv_path = os.path.join(here, "runs_summary.csv")
out_df.to_csv(csv_path, index=False)
print(f"Saved summary to {csv_path}")
