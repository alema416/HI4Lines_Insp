# Inference

This is the third step in the HI4Lines_Insp application: Inference

## Classifier Inference for Hardware Aware Optimization

With the client(s) training model instances and the servers pending to compile, the only part remaining is getting the relevant metrics from the edge device.

Go to the Inference directory and run the following:

```
tmux new-session -A -s server_raspi
source degirum_env/bin/activate
python3 server_raspi.py
```

Now the optimization pipeline is online. To see the progress from a client run:

```
optuna-dashboard postgresql://<username>:<password>@<deviceIP>/optuna_db --host 0.0.0.0 --port <port>
```
and access from: 

```
http://<deviceIP>:<port>
```
## Pipeline Inference for Evaluation

To run/evaluate the whole pipeline (detection+classification) per sample with respect to different offloading thresholds run:

```
python3 evpers.py
```
The evaluation code utilizes the following repo: https://github.com/sanchit2843/object_detection_metrics_calculation
