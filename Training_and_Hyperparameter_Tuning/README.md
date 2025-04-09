# Classifier Training/Hardware Aware Hyperparameter Tuning Process

This is the second step in the HI4Lines_Insp application: Classifier Training/Hardware Aware Hyperparameter Tuning Process

Our goal is to acquire a classifier which includes reliable failure prediction by confidence calibration. For this, we utilize as a starting point the code by this paper: https://github.com/Impression2805/FMFP 
And to select the best model we use the AUGRC metric as defined here : https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/main/torch_uncertainty/metrics/classification/risk_coverage.py

In order to determine the classifier model with the best confidence seperation **on the hardware of interest**, we run an extensive hyperparameter tuning experiment. 
Specifically, when each iteration of the model (which correspond to a set of hyperparameters) is trained to 200 epochs, we get a resulting pytorch state dictionary. This dictionary is automatically converted to an edge-device-compatible format,
which is then used for on-device evaluation. The final metric we get is the validation set hardware AUGRC, which is the variable we are trying to minimize in our experiment. In order to run this hardware aware hyperparameter optimization effectively, we designed it as a parallel procedure.
We utilize 2 GPU-powered systems which train the model with different hyperparameters and handle the compilation independently. We also utilize one edge device, with a software lock to avoid clashing in the case of simultaneous training completion. The hyperparameter optimization is handled
by an optuna experiment running off a postgreSQL database in the network. Note: more computational nodes can be added as needed. The procedure is clearly outlined by the following diagram:

![YOLO on EPRI-dataset](https://github.com/user-attachments/assets/ce918f12-e20e-452b-bf0a-22385f2001d7)
