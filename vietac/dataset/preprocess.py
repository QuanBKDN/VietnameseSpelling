import random
import re
import unicodedata
from multiprocessing import Pool
from typing import Any, Dict, List, NoReturn, Union

import pandas
import transformers
from tqdm import tqdm, trange
from vietac.dataset.clean import remove_repeat_char
from vietac.utils import logger


tokenizer = None


def multiprocess(func, iter_args, num_workers: int = 8, **kwargs):
    pool = Pool(processes=num_workers)
    jobs = [pool.apply_async(func, (arg,), kwargs) for arg in iter_args]

    results = [j.get() for j in tqdm(jobs)]
    pool.close()
    pool.join()

    return results


def encode(text: str) -> Union[Any, NoReturn]:
    """
    Encode text in to vector
    Args:
        text: (`str`) Text need to be encoded

    Returns: Union[Any, NoReturn] Vector represent for input text

    """
    global tokenizer
    if callable(tokenizer):
        return tokenizer(text, max_length=128, truncation=True, verbose=True)["input_ids"]
    else:
        raise TypeError


def create_training_data(
    dataframe: pandas.DataFrame, tokenizer_arg: transformers.AutoTokenizer, num_processes: int = 4
) -> Dict[str, List[Any]]:
    """
    Create data for training.
    Args:
        dataframe: (`pandas.DataFrame`) Text data.
        tokenizer_arg: (`transformers.AutoTokenizer`) Tokenizer used to encode text
        num_processes: (`int`) Number of processes (default=4)
    Returns: (`Dict[str, List[Any]]`) Data after encoded.

    """
    global tokenizer
    tokenizer = tokenizer_arg
    detect_prefix = "detection: "
    correct_prefix = "correction: "

    detect_idx = [i for i in range(len(dataframe))]
    random.shuffle(detect_idx)
    correct_idx = [i for i in range(len(dataframe))]
    random.shuffle(correct_idx)

    inputs = []
    labels = []
    input_text = dataframe["input_text"].to_list()
    detected = dataframe["detected"].to_list()
    corrected = dataframe["corrected"].to_list()
    logger.info("Adding prefix...")
    for i in trange(len(dataframe)):
        # detect_task
        inputs.append(detect_prefix + input_text[detect_idx[i]])
        labels.append(detected[detect_idx[i]])

        # correct_task
        if type(corrected[correct_idx[i]]) == str:
            inputs.append(correct_prefix + detected[correct_idx[i]])
            labels.append(corrected[correct_idx[i]])

    logger.info(f"Encoding inputs on {num_processes} processes...")
    input_ids = multiprocess(encode, inputs, num_workers=num_processes)
    del inputs
    logger.info(f"Encoding labels {num_processes} processes...")
    label_ids = multiprocess(encode, labels, num_workers=num_processes)
    del labels
    model_inputs = {
        "input_ids": input_ids,
        "labels": label_ids,
    }
    return model_inputs


def batching(
    text_list: List[str], batch_size: int = 32, apply_remove_duplicate: bool = False
) -> List[List[str]]:
    """
    Split a text list in to batch
    Args:
        text_list: (`List[str]`) Text list need to be batched
        batch_size: (`int`) Batch size
        apply_remove_duplicate: (`bool`) Apply remove duplicate chars option.

    Returns: (`List`) List of minibatch

    """
    if apply_remove_duplicate:
        text_list = [remove_repeat_char(text) for text in text_list]
    result_list = []
    data_len = len(text_list)
    for idx in range(int(data_len / batch_size) + 1):
        if (idx + 1) * batch_size >= data_len:
            end_offset = data_len
        else:
            end_offset = (idx + 1) * batch_size
        result_list.append(text_list[idx * batch_size : end_offset])
    if not result_list[-1]:
        return result_list[:-1]
    return result_list


def remove_unknown_char(text: str) -> str:
    """Remove some unknown characters"""
    text = text.replace("“", '"')
    text = text.replace("–", "-")
    text = text.replace("”", '"')
    return text


def prepare_inference_data(text: List[str]) -> List[str]:
    """
    Clean input text for inference step
    Args:
        text: (List[`str`]) - list of texts need to be clean

    Returns: (List[`str`]) - list of texts after cleaned

    """

    result = []
    for sentence in text:
        sentence = unicodedata.normalize("NFKC", sentence)
        sentence = remove_repeat_char(sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = remove_unknown_char(sentence)
        sentence = sentence.strip()
        result.append(sentence)
    return result
