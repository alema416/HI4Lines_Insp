# classification_eval.py: classification models evaluator
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#

import os
import inspect
import statistics
import degirum as dg
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from degirum_tools.eval_support import ModelEvaluatorBase
from degirum_tools.ui_support import Progress



class ImageClassificationModelEvaluator(ModelEvaluatorBase):
    """
    This class computes the Top-k Accuracy for Classification models.
    """

    def __init__(self, model: dg.model.Model, split: str, id: int, ide: str, **kwargs):
        """
        Constructor.

        Args:
            model (Detection model): PySDK classification model object
            kwargs (dict): arbitrary set of PySDK model parameters and the following evaluation parameters:
                show_progress (bool): show progress bar
                top_k (list) : List of `k` values in top-k, default:[1,5].
                foldermap (dict): mapping of model category IDs to image folder names.
                    For example: {0: "person", 1: "car"}
        """

        #
        # classification evaluator parameters:
        #

        # List of `k` values in top-k, default:[1,5].
        self.top_k: list = [1, 5]
        self.split = split
        self.id = id
        self.ide = ide
        # Mapping of model category IDs to image folder names.
        # For example: {0: "person", 1: "car"}
        self.foldermap: Optional[dict] = None

        if not model.output_postprocess_type == "Classification":
            raise Exception("Model loaded for evaluation is not a Classification Model")

        # base constructor assigns kwargs to model or to self
        super().__init__(model, **kwargs)

    @staticmethod
    def default_foldermap(folder_list: List[str]) -> Dict[int, str]:
        """
        Constructs a default foldermap from the folder list:
        a key is just an index of the corresponding folder.
        """
        return {i: folder for i, folder in enumerate(folder_list)}

    def evaluate(
        self,
        image_folder_path: str,
        ground_truth_annotations_path: str,
        max_images: int = 0,
    ) -> list:
        """
        Evaluation for the classification model.

        Args:
            image_folder_path (str): Path to images
            ground_truth_annotations_path (str): not used
            max_images (int): not used

        Returns:
            2-element list: Top-k accuracy and per-class accuracy statistics.
        """

        #
        # initialization
        #

        folder_list = sorted(os.listdir(image_folder_path))
        if self.foldermap is None:
            self.foldermap = self.default_foldermap(folder_list)

        total_correct_predictions = [
            [0] * len(self.foldermap) for _ in range(len(self.top_k))
        ]
        images_in_folder = []
        total_images_in_folder = []
        all_per_class_accuracies = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        # scan folders for images
        for folder_idx, category_folder in enumerate(self.foldermap.values()):
            image_dir_path = Path(image_folder_path) / category_folder

            all_images = [
                str(image_path)
                for image_path in image_dir_path.glob("*")
                if image_path.suffix.lower() in image_extensions
            ]
            images_in_folder.append(all_images)
            total_images_in_folder.append(len(all_images))

        total_images = sum(total_images_in_folder)

        #
        # evaluation loop
        #

        max_images = min(max_images, total_images) if max_images > 0 else total_images
        if self.show_progress:
            progress = Progress(max_images)

        processed_images = 0
        tr = 0
        fa = 0
        tr_conf = []
        fa_conf = []
        FILE_CNF = []
        FILE_LBL = []
        rows = []
        for folder_idx, category_folder in enumerate(self.foldermap.values()):
          
            per_class_accuracies = [-1.0] * len(self.top_k)
            processed_images_in_class = 0
            #print(images_in_folder[folder_idx])
            for image_path, predictions in zip(images_in_folder[folder_idx], self.model.predict_batch(images_in_folder[folder_idx])):
            #for predictions in self.model.predict_batch(images_in_folder[folder_idx]):
                tmp_score = predictions.results[0]['score']
                # Iterate over each top_k value
                for k_i, k in enumerate(self.top_k):
                    # Sort predictions and get top-k results
                    sorted_predictions = sorted(
                        predictions.results, key=lambda x: x["score"], reverse=True
                    )[:k]
                    #print(f"{sorted_predictions[0]['score']} --> {sorted_predictions[-1]['score']}")
                    
                    
                    top_categories = [
                        int(pred["category_id"]) for pred in sorted_predictions
                    ]
                    
                    top_classes = [
                        self.foldermap[top]
                        for top in top_categories
                        if top in self.foldermap
                    ]
                    # Check if ground truth is in top-k predictions
                    if category_folder in top_classes:
                        total_correct_predictions[k_i][folder_idx] += 1
                        tr += 1
                        tr_conf.append(tmp_score)
                        
                    else:
                        fa += 1
                        fa_conf.append(tmp_score)
                FILE_CNF.append(tmp_score)
                FILE_LBL.append(1 if category_folder in top_classes else 0)
                other = 'healthy' if category_folder == 'broken' else 'broken'
                row = {'filename': os.path.basename(image_path), 'confidence': tmp_score, 'prediction': category_folder if category_folder in top_classes else other, 'ground_truth': category_folder, 'correct': 1 if category_folder in top_classes else 0}
                rows.append(row)
                #print(f'{predictions}')# {tmp_score} {}')
                processed_images_in_class += 1
                processed_images += 1
                
                
                per_class_accuracies = [
                    total_correct_predictions[k_i][folder_idx]
                    / processed_images_in_class
                    for k_i, _ in enumerate(self.top_k)
                ]

                if self.show_progress:
                    accuracy_str = f"{category_folder}: " + ", ".join(
                        [
                            f"top{k} = {per_class_accuracies[i] * 100:.1f}%"
                            for i, k in enumerate(self.top_k)
                        ]
                    )
                    progress.step(message=accuracy_str)
                
                if processed_images >= max_images:
                    all_per_class_accuracies.append(per_class_accuracies)
                    
                    break
            
            else:  # for predictions... loop completes all iterations

                all_per_class_accuracies.append(per_class_accuracies)
                continue
            break
        print(f'{tr} success, {fa} error')
        #print(f"SUCCESS - mean: {statistics.mean(tr_conf)}, std: {statistics.stdev(tr_conf)}")
        if len(fa_conf) >= 2:
            std_err = statistics.stdev(fa_conf)
            mean_err = statistics.mean(fa_conf)
        else:
            std_err = 0.0
            mean_err = 0.0
        
        if len(tr_conf) >= 2:
            std_tr = statistics.stdev(tr_conf)
            mean_tr = statistics.mean(tr_conf)
        else:
            std_tr = 0.0
            mean_tr = 0.0
        print(f"SUCCESS - mean: {mean_tr}, std: {std_tr}")
        print(f"ERROR - mean: {mean_err}, std: {std_err}")

        accuracies = [
            sum(total_correct_predictions[k_i]) / processed_images
            for k_i, _ in enumerate(self.top_k)
        ]

        # show final accuracy
        if self.show_progress:
            accuracy_str = ", ".join(
                [
                    f"top{k} = {accuracies[i] * 100:.1f}%"
                    for i, k in enumerate(self.top_k)
                ]
            )
            progress.message = accuracy_str
        with open(f'{self.ide}labels_{self.id}_{self.split}.txt', "w") as file:
            for line in FILE_LBL:
                file.write(str(line) + "\n")  # Adding newline character
        with open(f'{self.ide}confs_{self.id}_{self.split}.txt', "w") as file:
            for line in FILE_CNF:
                file.write(str(line) + "\n")  # Adding newline character
        df = pd.DataFrame(rows)
        df.to_csv(f'per_sample_{self.split}.csv', index=False)
        return [accuracies, all_per_class_accuracies], mean_tr, std_tr, mean_err, std_err, tr, fa
