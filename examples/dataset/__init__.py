

import logging
import pickle 
def load_dataset(dataset_id):
    if dataset_id == '20230321':
        with open("./dataset/dataset_20230321_2500.pkl", "rb") as f:
            dataset = pickle.load(f)
    elif dataset_id == '20230417':
        with open("dataset/dataset_20230417_4200.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        logging.exception("daraset not exist", dataset_id)
        # raise Exception("daraset not exist", dataset_id)
    
    return dataset