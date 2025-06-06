import pandas as pd

for device in ["HAILO", "ST"]:
    print(f'{device}')
    id = 22 if device == 'ST' else 14
    for split in ['train', 'val', 'test']:
        print(f'{split}')
        # 1) Load your data. 
        #    If your Google Sheet has columns: filename, confidence, prediction, ground_truth, correct
        #    first “Download as … CSV” (File → Download → Comma-separated values) and put that filename here:
        df = pd.read_csv(f"/home/alema416/dev/work/HI4Lines_Insp/deploy/rpi/deploy/data/optimization_report - MODEL_{id}_{device}_per_sample_{split}.csv")  

        # 2) Make sure your labels match exactly, e.g. “broken” vs “healthy” (case‐sensitive).
        #    If you use different names, just substitute them below. 
        if device == 'HAILO':
            POS = "broken"
            NEG = "healthy"
        else:
            POS = 0
            NEG = 1
        
        # 3) Compute TP, FP, TN, FN by comparing prediction vs ground_truth
        tp = ((df["prediction"] == POS) & (df["ground_truth"] == POS)).sum()
        fp = ((df["prediction"] == POS) & (df["ground_truth"] == NEG)).sum()
        tn = ((df["prediction"] == NEG) & (df["ground_truth"] == NEG)).sum()
        fn = ((df["prediction"] == NEG) & (df["ground_truth"] == POS)).sum()

        # 4) Check that counts make sense
        print(f"TP = {tp}, FP = {fp}, TN = {tn}, FN = {fn}")
        print(f"Total positives (actual) = {tp + fn}, total negatives (actual) = {fp + tn}")

        # 5) Compute FPR and TNR
        #    (Make sure denominator (FP+TN) is not zero.)
        den_neg = fp + tn
        if den_neg > 0:
            fpr = fp / den_neg
            tnr = tn / den_neg
            print(f"FPR = {fpr:.4f}  ({fpr*100:.1f}%)")
            print(f"TNR = {tnr:.4f}  ({tnr*100:.1f}%)")
        else:
            print("No actual negatives found (FP + TN = 0), cannot compute FPR/TNR.")
