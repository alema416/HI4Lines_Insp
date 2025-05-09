#!/usr/bin/env python3
import os
import pandas as pd
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

# 6) Build DataFrame and sort by start time
df = pd.DataFrame(rows)
df = df.sort_values("started")

# … after you build & sort df …
'''
# drop any run that has NaN in **any** of the columns we care about
df = df.dropna(subset=[
    "train_acc","val_acc","test_acc",
    "acc_hw_train","acc_hw_val","acc_hw_test",
    "augrc_hw_train","augrc_hw_val","augrc_hw_test"
], how="any")
'''
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

'''
# 7) Print one big Markdown table without the Parent column
print(df[[
    "run_name",
    "train_acc","val_acc","test_acc",
    "acc_hw_train","acc_hw_val","acc_hw_test",
    "augrc_hw_train","augrc_hw_val","augrc_hw_test"
]].to_markdown(
    index=False,
    headers=[
      "run_name",
      "train_acc","val_acc","test_acc",
      "acc_hw_train","acc_hw_val","acc_hw_test",
      "augrc_hw_train","augrc_hw_val","augrc_hw_test"
    ]
))
'''
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
