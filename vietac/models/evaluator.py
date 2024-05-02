import re
from typing import Dict, NoReturn

from pandas import DataFrame
from tqdm import trange
from vietac.dataset.clean import clean_input_text, normalize_detection, remove_punc
from vietac.models import Corrector


class Evaluator:
    def __init__(self, model_path: str = None):
        """
        Create an Evaluator
        Args:
            model_path: (`str`, *optional*) - Path to model weights and configs
        """
        self.corrector = None
        self.data = DataFrame()
        self.prediction = {}
        if model_path is not None:
            self.corrector = Corrector(model_path)

    def get_f1_score(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate F1-Score

        Returns: (`dict[str, dict[str, float]]`) Return F1-Score on dictionary format
        """
        if self.corrector is None and self.data is None:
            raise "Model and data are None"
        inference = self.corrector.infer(self.data["error_sentence"].to_list())
        self.prediction["detection"], self.prediction["correction"] = inference
        return Evaluator.f1_score(
            self.data["detection_target"].to_list(),
            self.prediction["detection"],
            self.data["correction_target"].to_list(),
            self.prediction["correction"],
        )

    def set_eval_data(self, data: DataFrame, column_map: Dict[str, str]) -> NoReturn:
        """
        Set eval data for Evaluator
        Args:
            data: (`pandas.DataFrame`) Evaluation data
            column_map: (`dict[str, str]`) map to read neccesary columns by name
        """
        self.data["error_sentence"] = [
            clean_input_text(i) for i in data[column_map["error_sentence"]].to_list()
        ]
        self.data["detection_target"] = [
            clean_input_text(i) for i in data[column_map["detection_target"]].to_list()
        ]
        self.data["correction_target"] = [
            clean_input_text(i) for i in data[column_map["correction_target"]].to_list()
        ]

    @staticmethod
    def f1_score(
        detection_target: list,
        detection_predict: list,
        correction_target: list,
        correction_predict: list,
    ) -> Dict[str, Dict[str, float]]:
        """

        Args:
            detection_target: (`list[str]`) - Target on error detection task
            detection_predict: (`list[str]`) - Prediction of model on error detection task
            correction_target: (`list[str]`) - Target on error correction task
            correction_predict: (`list[str]`) - Prediction of model on error correction task

        Returns: `dict[str, dict[str, float]` - F1-score on evaluation data

        """

        detection_predict = normalize_detection(detection_predict)
        num_true_detections = 0
        num_true_corrections = 0
        num_errors_predict = 0
        num_actual_errors = 0

        for idx in trange(len(detection_predict)):
            if idx == 44:
                print("X")
            detection = Evaluator.count_error(detection_target[idx], detection_predict[idx])
            num_actual_errors += detection["actual_errors"]
            num_true_detections += detection["true_predict"]
            num_errors_predict += detection["predict_errors"]

            correction = Evaluator.count_correction(
                correction_target[idx], correction_predict[idx], detection_predict[idx]
            )
            num_true_corrections += correction["true_correction"]

        d_precision = num_true_detections / num_errors_predict
        d_recall = num_true_detections / num_actual_errors
        d_f1 = 2 * (d_precision * d_recall) / (d_precision + d_recall)

        c_precision = num_true_corrections / num_errors_predict
        c_recall = num_true_corrections / num_actual_errors
        c_f1 = 2 * (c_precision * c_recall) / (c_precision + c_recall)
        return {
            "detection-task": {"precision": d_precision, "recall": d_recall, "f1": d_f1},
            "correction-task": {"precision": c_precision, "recall": c_recall, "f1": c_f1},
        }

    @staticmethod
    def count_error(detection_target: str, detection_predict: str) -> Dict[str, int]:
        """
        Count number of actual errors, predict errors and true predictions
        Args:
            detection_target: (`str`) Detection target
            detection_predict: (`str`) Detection predict

        Returns: (`Dict[str, int]`) - return number of actual errors, predict errors and true predictions

        """
        errors_predict = re.findall(r"<error>.+?</error>", detection_predict)
        errors_target = re.findall(r"<error>.+?</error>", detection_target)
        num_actual_errors = len(errors_target)
        num_predict_errors = len(errors_predict)
        difference = num_predict_errors - num_actual_errors
        num_true_predict = 0
        offset = 0
        for prediction in errors_predict:
            for i, true_error in enumerate(errors_target[offset:]):
                if prediction == true_error:
                    num_true_predict += 1
                    offset = i
                    break
        if difference > 0:
            num_actual_errors += difference
        return {
            "actual_errors": num_actual_errors,
            "predict_errors": num_predict_errors,
            "true_predict": num_true_predict,
        }

    @staticmethod
    def count_correction(
        correct_sentence: str,
        predict_sentence: str,
        detect_sentence: str,
        different_len_limit: int = 2,
    ) -> Dict[str, int]:
        """
        Count number of true predictions
        Args:
            correct_sentence: (`str`) - Target on error, correction task
            predict_sentence: (`str`) - Prediction on error correction task
            detect_sentence: (`str`) - Prediction on error detection task
            different_len_limit: (`int`) - Limit of difference len

        Returns: (`Dict[str, int]`) - Number of true predictions

        """
        if len(correct_sentence.split()) != len(predict_sentence.split()):
            print("x")
        # correct_sentence = correct_sentence.replace("\n", "")
        # detect_sentence = detect_sentence.replace("</s>", "")
        # while "<pad>" in detect_sentence:
        #     detect_sentence = detect_sentence.replace("<pad>", "")
        offset = 0
        true_correction = 0
        predict_sentence = remove_punc(predict_sentence)
        correct_sentence = remove_punc(correct_sentence)
        detect_sentence = remove_punc(detect_sentence)
        errors = re.findall(r"<error>.+?</error>", detect_sentence)

        for error in errors:
            detect_sentence = detect_sentence.replace(error, "")
        outside_words = detect_sentence.split()

        predict_sentence_t = predict_sentence.split()
        correct_sentence_t = correct_sentence.split()
        for word in outside_words:
            try:
                correct_sentence_t.remove(word)
                predict_sentence_t.remove(word)
            except ValueError:
                if word == "<error>":
                    print(word)

                print(word)

        predict_words = predict_sentence_t
        correct_words = correct_sentence_t

        if len(predict_words) == len(correct_words):
            for i, predict_word in enumerate(predict_words):
                if predict_word == correct_words[i]:
                    true_correction += 1
            return {"true_correction": true_correction}
        if abs(len(predict_words) - len(correct_words)) > different_len_limit:
            return {"true_correction": 0}
        for i, predict_word in enumerate(predict_words):
            for j, correct_word in enumerate(correct_words[offset:]):
                if correct_word == predict_word:
                    if (offset - i) > 2:
                        return {"true_correction": true_correction}
                    else:
                        true_correction += 1
        if true_correction > len(errors):
            print(correct_sentence)
            print(predict_sentence)
            print(detect_sentence)
        return {"true_correction": true_correction}
