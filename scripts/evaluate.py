import argparse
import os

import pandas as pd
from vietac.models import Evaluator
from vietac.utils.configs import read_config


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="vietac/configs/configs.yaml")
args = parser.parse_args()

if __name__ == "__main__":
    config = read_config(args.config_path)
    column_map = {
        "error_sentence": "augmented",
        "detection_target": "label",
        "correction_target": "original",
    }
    data_dir = config.evaluate.data_path
    evaluator = Evaluator(model_path=config.evaluate.model_path)

    errors_type = os.listdir(data_dir)
    for error_type in errors_type:
        eval_data_files = os.listdir(os.path.join(data_dir, error_type))
        logs_path = config.evaluate.log_path
        with open(os.path.join(logs_path, error_type), "w") as f:
            for file_name in eval_data_files:
                print("Evaluating on", file_name)
                eval_data_path = os.path.join(data_dir, error_type, file_name)
                eval_data = pd.read_csv(eval_data_path)
                evaluator.set_eval_data(data=eval_data, column_map=column_map)
                f.write("{e}% error: \n".format(e=file_name[-9:-7]))
                r = str(evaluator.get_f1_score())
                print(r)
                f.write(r)
                f.write("\n")
