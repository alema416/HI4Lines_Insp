from mlflow.tracking import MlflowClient
import os
client = MlflowClient(tracking_uri=f"file://{os.path.abspath(mlruns)}")

# Check a few runs (e.g., first 5)
for run in all_runs[:5]:
    print(f"\nRun ID: {run.info.run_id}")
    metrics = client.get_metric_history(run.info.run_id, "val_acc")
    if len(metrics) > 1:
        print(f"Run {run.info.run_id} has per-epoch 'val_acc' metrics:")
        for m in metrics:
            print(f"  Step {m.step}: val_acc = {m.value}")
    else:
        print(f"Run {run.info.run_id} does NOT have per-epoch 'val_acc' metrics.")
