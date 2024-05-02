import os.path
import pickle
import re
import shutil
import string
from copy import copy
from typing import Dict, List, Tuple, Union

import numpy
import torch
from Levenshtein import ratio
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from vietac.dataset.augment_error import telex_transform_base
from vietac.dataset.clean import (
    clean_error_tag,
    normalize_detection,
    remove_duplicate_space,
    remove_exception_error_tag,
    remove_punc,
    remove_repeat_char,
)
from vietac.dataset.preprocess import batching, prepare_inference_data
from vietac.utils import logger
from vietac.utils.postprocessing import get_corrected_word, matching, process_unknown_token
from vietac.utils.preprocessing import extract_unprocessed_sample, pre_correct


VIETAC_DICTIONARIES_URL = (
    "http://minio.dev.ftech.ai/vietac-dictionaries-v1.0.1-bc7baf19/dictionaries.zip"
)
VIETAC_MODEL_URL = "http://minio.dev.ftech.ai/vietac-model-v3.0-51aad0e5/vietac_model.zip"
VIETAC_DICTIONARIES = "dictionaries.zip"
VIETAC_MODEL = "vietac_model.zip"


class Corrector:
    def __init__(
        self,
        model_path: str = None,
        dictionary_path: str = None,
        download_dictionary: bool = False,
        download_model: bool = False,
        model_correction_threshold: float = 0.7,
        matching_threshold: float = 0.5,
        levenshtein_threshold: float = 0.87,
        seq_max_length: int = 128,
        device=None,
    ):
        """
        Create a Corrector
        Args:
            model_path: (`str`) - Path to model weights, tokenizer and configs
            dictionary_path: (`str`) - Path to telex dictionary
            download_dictionary: (`bool`) - Download the dictionary, save to dictionary_path.
            model_correction_threshold: (`float`) - T5 will refuse to predict if Error ratio in a sentence is more than this threshold.
            matching_threshold: (`float`) - Skip predict word if word matching score less than this threshold.
            levenshtein_threshold: (`float`) - Skip predict word if word matching score less than this threshold.
            seq_max_length: (`int`) - The maximum sequence length.
            device: (`str`) - Device to run model
        """
        if dictionary_path is None:
            dictionary_path = "/tmp/vietac_dictionaries"
            logger.warning("dictionary_path is not specified, default at /tmp/vietac_dictionaries")
        if model_path is None:
            model_path = "/tmp/vietac_model"
            logger.warning("model_path is not specified, default at /tmp/vietac_model")

        if download_dictionary:
            self.download_dictionaries(dictionary_path)
        if download_model:
            self.download_model(model_path)

        try:
            with open(os.path.join(dictionary_path, "telex_dictionary.pack"), "rb") as f:
                self.telex_map = pickle.load(f)

            self.vocab_corrector = VocabCorrector(
                telex_map_path=os.path.join(dictionary_path, "telex_wordgroup_map.pkl"),
                right_word_map_path=os.path.join(dictionary_path, "right_key_map.pkl"),
                left_word_map_path=os.path.join(dictionary_path, "left_key_map.pkl"),
                threshold=levenshtein_threshold,
            )
        except FileNotFoundError as e:
            logger.warning(
                "Dictionaries not found - Set `True` on `download_dictionary` to download"
            )
            raise e

        self.max_length = seq_max_length
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = (
                torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        except OSError as e:
            logger.warning("Model not found - Set `True` on `download_model` to download")
            raise e
        self.model.to(self.device)
        self.model_correction_threshold = model_correction_threshold
        self.model_matching_threshold = matching_threshold

    @staticmethod
    def download_dictionaries(save_path):
        os.system(f"wget --progress=bar:force:noscroll {VIETAC_DICTIONARIES_URL} -P {save_path}")
        shutil.unpack_archive(filename=f"{save_path}/{VIETAC_DICTIONARIES}", extract_dir=save_path)
        print("Download finished!")

    @staticmethod
    def download_model(save_path):
        os.system(f"wget --progress=bar:force:noscroll {VIETAC_MODEL_URL} -P {save_path}")
        shutil.unpack_archive(filename=f"{save_path}/{VIETAC_MODEL}", extract_dir=save_path)
        print("Download finished!")

    def correct_telex_by_word(self, word: str) -> str:
        """
        Correct telex-error word by dictionary
        Args:
            word: `str` - The word need to be correct
        Returns:
            `str` - The word after corrected
        """
        punc = ""
        if word[-1] in string.punctuation:
            punc = word[-1]
            word = word[:-1]
        try:
            return self.telex_map[word] + punc
        except KeyError:
            return word

    def correct_telex_by_sentence(self, sentence: str) -> str:
        """
        Correct telex-error sentence by dictionary
        Args:
            sentence: (`str`) - The sentence need to be corrected
        Returns:
            `str` - The sentence after corrected
        """
        result = []
        word_list = sentence.split()
        for word in word_list:
            result.append(self.correct_telex_by_word(word))
        return " ".join(result)

    def correct_telex_by_list(self, text_list: List[str]) -> List[str]:
        """
        Correct telex errors on a list of sentences
        Args:
            text_list: (List[`str`]) - List of sentences need to be corrected.

        Returns: (List[`str`]) - List of sentences after corrected telex errors.

        """
        result = []
        for text in text_list:
            result.append(self.correct_telex_by_sentence(text))
        return result

    def _tokenize_batch(self, text_list: List[str], prefix: str) -> torch.Tensor:
        """
        Encode a texts list to tensors
        Args:
            text_list: (`List[str]`) - List of texts need to be tokenized
            prefix: (`str`) - Task prefix
        Returns: (`torch.Tensor`) - input_ids of text_list

        """
        text_list = [prefix + sentence for sentence in text_list]
        return self.tokenizer.batch_encode_plus(
            text_list,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).to(self.device)

    def _decode_batch(
        self, vectors_list: List[numpy.array], skip_special_tokens: bool = False
    ) -> List[str]:
        """
        Decode model outputs into strings
        Args:
            vectors_list: (`list`) - List of model outputs
            skip_special_tokens: (`bool`, *optional*) - Keep/Remove special token. Default: False

        Returns: (`list`) - List of string after decoded

        """
        results = []
        for vector in vectors_list:
            results.append(
                self.tokenizer.decode(
                    vector.detach().cpu().numpy(), skip_special_tokens=skip_special_tokens
                )
            )
        return results

    def _detect_errors(self, texts: Union[str, list]) -> List[str]:
        """
        Detect error in a text or a list of texts
        Args:
            texts: (`str` or `list[str]`) - Text need to be detected

        Returns: (`str` or `list[str]`) - Text with error tag af ter detected

        """
        if type(texts) == str:
            texts = [texts]
        model_inputs = self._tokenize_batch(texts, prefix="detection: ")
        error_detected = self.model.generate(
            model_inputs["input_ids"].to(self.device), max_length=self.max_length
        )
        error_detected = self._decode_batch(error_detected, skip_special_tokens=False)
        torch.cuda.empty_cache()
        return error_detected

    def _correct_errors(self, detection: List[str]) -> List[str]:
        """
        Correct errors on a list of texts
        Args:
            detection: (`list[str]`) A list of text with error tag

        Returns: (`list[str]`) A list of text after corrected

        """
        model_inputs = self._tokenize_batch(detection, prefix="correction: ")
        error_corrected = self.model.generate(
            model_inputs["input_ids"].to(self.device), max_length=self.max_length
        )
        text_corrected = self._decode_batch(error_corrected, skip_special_tokens=True)
        for i, sent in enumerate(detection):
            error_ratio = sent.count("<error>") / len(text_corrected[i].split())
            if error_ratio > self.model_correction_threshold:
                sent = list(clean_error_tag([sent]))[0]
                text_corrected[i] = sent
        torch.cuda.empty_cache()
        return text_corrected

    def infer(
        self, input_text: Union[str, list], batch_size: int = 16
    ) -> Union[List[str], Tuple[List[str], List]]:
        """


        Correct a sentence or a list of sentences
        Args:
            input_text: (`str` or `list[str]`) A string or list of string need to be corrected.
            batch_size: (`int`, *optional*) Decrease execution time if text is a list.
        Returns: (`Tuple`) a tuple included prediction on both error detection and error correction tasks.
        """
        if isinstance(input_text, str):
            input_text = [input_text]
        # Extract unprocessed samples which have any annotation or formulas
        # These samples will not be corrected by VietAC
        input_text = prepare_inference_data(input_text)
        text, unprocessed_index = extract_unprocessed_sample(text_list=input_text)

        # Start pre-correct input
        text = pre_correct(text)
        input_data = batching(text_list=text, batch_size=batch_size, apply_remove_duplicate=False)
        detection = []
        correction = []
        precorrection = []

        # Start correcting input by model
        for batch in tqdm(input_data):
            batch = self.correct_telex_by_list(batch)
            batch_detection = self._detect_errors(batch)
            batch_precorrection = self.vocab_corrector.correct_by_list(
                detected_list=batch_detection
            )
            detection.extend(batch_detection)
            precorrection.extend(batch_precorrection)
            correction.extend(self._correct_errors(batch_precorrection))
            torch.cuda.empty_cache()
            # Merge unprocessed samples in to results
        correction = matching(
            detection_list=precorrection,
            correction_list=correction,
            matching_threshold=self.model_matching_threshold,
        )

        # Put the unprocessed samples into the output
        if unprocessed_index:
            for idx in unprocessed_index:
                correction.insert(idx, input_text[idx])

        correction = process_unknown_token(correction, input_text)

        # The origin text must be removed punc before get_corrected
        # because get_corrected will process some removed standalone punc as a corrected error
        # Note that unprocessed_sample will not pass to get_corrected
        errors = [
            get_corrected_word(origin_text=remove_punc(input_text[i]), corrected_text=correct_text)
            if correct_text != input_text[i]
            else []
            for i, correct_text in enumerate(correction)
        ]

        return correction, errors


class VocabCorrector:
    def __init__(self, telex_map_path, right_word_map_path, left_word_map_path, threshold=0.8):
        with open(telex_map_path, "rb") as f:
            self.map = pickle.load(f)

        with open(right_word_map_path, "rb") as f:
            self.right_word_map = pickle.load(f)

        with open(left_word_map_path, "rb") as f:
            self.left_word_map = pickle.load(f)

        self.black_list = ["<error>", "</error>"]
        self.threshold = threshold

    def get_similar_scores(self, word: str, key: str = "none") -> Dict[str, float]:
        """
        Calculate Levenshtein distance
        Args:
            word: (`str`) Error word group
            key: (`str`) The correct side in error word group

        Returns: (Dict[`str`,`float`]) A dictionary contains all suggest word groups and following scores.

        """
        if key == "none":
            return {w: ratio(word, w) for w in self.map.keys()}
        elif key == "right":
            left_word, right_word = word.split("_")
            try:
                return {w: ratio(left_word, w) for w in self.right_word_map[right_word]}
            except KeyError:
                return {left_word: 0}
        elif key == "left":
            left_word, right_word = word.split("_")
            try:
                return {w: ratio(right_word, w) for w in self.left_word_map[left_word]}
            except KeyError:
                return {right_word: 0}

    def predict_group(self, word_group: str, key: str) -> Tuple[str, float]:
        """
        Returns suggested word group for error word group
        Args:
            word_group: (`str`) - Error word group
            key: The correct side in error word group

        Returns: (Tuple[`str`, `float`]) A tuple contains suggested word group and following score

        """
        scores = self.get_similar_scores(word_group, key)
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        predict_word, score = scores[0]
        if score >= self.threshold:
            if predict_word in self.map and key == "none":
                return self.map[predict_word], score
            elif key == "right":
                predict_group = "_".join(
                    [predict_word, self.telex_transform(word_group.split("_")[1])]
                )
                return self.map[predict_group], score
            elif key == "left":
                predict_group = "_".join(
                    [self.telex_transform(word_group.split("_")[0]), predict_word]
                )
                return self.map[predict_group], score
        else:
            return word_group, 0

    def extract_word_group(self, sentence: str) -> List[Dict[str, str]]:
        """
        Extract all word groups for each error word
        Args:
            sentence: (`str`) - Output from Error Detection module <error>|</error>

        Returns: (List[Dict[`str`, `str`]]) - A list contains all error word groups existed in sentence

        """
        word_group = []
        word_list = sentence.split()

        for i, word in enumerate(word_list):
            previous_word = ""
            next_word = ""
            is_previous_error = False
            is_next_error = False
            if word not in self.black_list and word_list[i - 1] == "<error>":
                j = i - 1
                while j >= 0:
                    if word_list[j] not in self.black_list:
                        previous_word = word_list[j]
                        if j == i - 3:
                            is_previous_error = True
                        break
                    else:
                        j -= 1
                j = i + 1
                while j < len(word_list):
                    if word_list[j] not in self.black_list:
                        next_word = word_list[j]
                        if j == i + 3:
                            is_next_error = True
                        break
                    else:
                        j += 1

            if previous_word != "":
                if not is_previous_error:
                    word_group.extend([{"group": "_".join([previous_word, word]), "key": "left"}])
                else:
                    word_group.extend([{"group": "_".join([previous_word, word]), "key": "none"}])

            if next_word != "":
                if not is_next_error:
                    word_group.extend([{"group": "_".join([word, next_word]), "key": "right"}])
                else:
                    word_group.extend([{"group": "_".join([word, next_word]), "key": "none"}])
        return word_group

    @staticmethod
    def pick_best_group(group_list: List[Tuple[str, float]]) -> str:
        """
        Pick the best candidate by similarity
        Args:
            group_list: (List[Tuple[`str`, `float`]]) - List of candidates contains suggested word and following score

        Returns: (`str`) - The best candidates

        """
        sorted_x = sorted(group_list, key=lambda x: x[1], reverse=True)
        return sorted_x[0][0]

    def correct_sentence(self, sentence: str) -> str:
        """
        Correct a sentence by vocabulary
        Args:
            sentence: (`str`) - Output from Error Detection module <error>|</error>

        Returns: (`str`) - Return a sentence after corrected

        """
        result = {}
        sentence = normalize_detection([sentence])[0]
        origin_sentence = copy(sentence)
        word_groups = self.extract_word_group(sentence)
        sentence = sentence.replace("<error>", "")
        sentence = sentence.replace("</error>", "")
        sentence = remove_duplicate_space(sentence)
        for group in word_groups:
            if group["key"] == "right":
                telex_group = "_".join(
                    [
                        self.telex_transform(group["group"].split("_")[0]),
                        group["group"].split("_")[1],
                    ]
                )

            elif group["key"] == "left":
                telex_group = "_".join(
                    [
                        group["group"].split("_")[0],
                        self.telex_transform(group["group"].split("_")[1]),
                    ]
                )
            else:
                telex_group = "_".join(
                    [self.telex_transform(word) for word in group["group"].split("_")]
                )

            prediction = self.predict_group(word_group=telex_group, key=group["key"])
            if prediction[0] != telex_group:
                origin_group = " ".join(group["group"].split("_"))
                group_predicted = " ".join(prediction[0].split("_"))
                if origin_group in result:
                    result[origin_group].append((group_predicted, prediction[1]))
                else:
                    result[origin_group] = [(group_predicted, prediction[1])]
        for origin_group in result:
            sentence = sentence.replace(origin_group, self.pick_best_group(result[origin_group]))
        return self.re_tag(origin_sentence, sentence.strip())

    def re_tag(self, sentence_before: str, sentence_after: str) -> str:
        """
        Re-tagging <error>|</error> token into corrected sentence for remaining error words.
        Args:
            sentence_before: (`str`) - Sentence before run correction
            sentence_after: (`str`) - Sentence after corrected

        Returns: (`str`) - Sentence with <error>|</error> token for remaining error words.

        """
        word_index = -1
        for word in sentence_before.split():
            if word not in self.black_list:
                word_index += 1
                if word != sentence_after.split()[word_index]:
                    tmp_sentence = sentence_before.replace(
                        " ".join([self.black_list[0], word, self.black_list[1]]),
                        sentence_after.split()[word_index],
                    )
                    if tmp_sentence == sentence_before:
                        tmp_sentence = sentence_before.replace(
                            word, sentence_after.split()[word_index]
                        )
                    sentence_before = tmp_sentence
        return sentence_before

    def correct_by_list(self, detected_list: List[str]) -> List[str]:
        """
        Run correction on list
        Args:
            detected_list: List of error detected sentences

        Returns: List for error corrected sentence

        """
        result = []
        for sentence in detected_list:
            text = self.correct_sentence(sentence)
            result.append(
                remove_repeat_char(remove_exception_error_tag(text), keep_duplicate_vowels=False)
            )
        return result

    @staticmethod
    def telex_transform(word: str) -> str:
        """
        Transform a Vietnamese word in to telex form
        Args:
            word: (`str`) - Word need to be transformed in to telex

        Returns: (`str`) - Word in telex form

        """
        words = []
        word, _ = telex_transform_base(word)
        try:
            diacritic = re.findall(r"-\w", word)
            word = word.replace(diacritic[0], "")
            word = word + diacritic[0][-1]
            words.append(word)

        except IndexError:
            words.append(word)
        results = []
        results.extend(words)
        return list(set(results))[0]
