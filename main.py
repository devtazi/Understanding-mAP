from datasets import load_dataset
from prediction import simulate_prediction
from utils import calculate_map
import numpy as np
import config

def main():
   dataset = load_dataset("HichTala/coco", streaming=True)
   iteration = iter(dataset['train'])
   prediction, total_real_objects = simulate_prediction(iteration)
   map = calculate_map(prediction, total_real_objects)
   
   print("\n" + "=" * 40)
   print("RÉSULTATS FINAUX DE LA SIMULATION")
   print(f"Images simulées        : {config.NOMBRES_IMAGES_SIMULEES}")
   print(f"Seuil IoU              : {config.SEUIL_IOU}")
   print("-" * 40)

   print("-" * 40)
   print(f"mAP : {map:.4f} ({map*100:.2f}%)")
   
    
if "__name__" == "__main__.py":
    main()