import re
import string
from copy import copy
from typing import Dict, List, Union

from pyvi import ViUtils
from tqdm import tqdm
from vietac.utils.constants import NOTATIONS, VOWELS, WORD_CHAR


def check_url(text: str) -> bool:
    """
    Check if "http" is in text.
    Args:
        text: (`str`) The text need to be checked

    Returns: True if "http" in text, otherwise.

    """
    return "http" in text


def check_len(text: str, min_len: int = 5) -> bool:
    """
    Check if len of input text less than min_len
    Args:
        text: (`str`) The text need to be checked.
        min_len: (`int`) Minimum len

    Returns: True if len(text) < min_len, otherwise.

    """
    return len(text.split(" ")) < min_len


def check_accents(text: str) -> bool:
    """
    Check if a text is not Vietnamese
    Args:
        text: (`str`) The text need to be checked.

    Returns: True if text has Vietnamese accents, otherwise.

    """
    return ViUtils.remove_accents(text).decode() == text


def check_punc(text: str) -> bool:
    """
    Check if a text has many punctuations
    Args:
        text: (`str`) The text need to be checked.

    Returns: True if number of punctuations in the input text more than 5, otherwise.

    """
    new_text = text.translate(str.maketrans(" ", " ", string.punctuation))
    return len(text) - len(new_text) > 5


def filter_data(raw_data_path: str, clean_data_path: str):
    """
    Clean raw text file.
    Args:
        raw_data_path: (`str`) Path to raw data
        clean_data_path: (`str`) Path to save clean data

    """
    with open(raw_data_path, "r") as raw:
        raw_texts = raw.readlines()
    with open(clean_data_path, "w") as clean:
        for text in tqdm(raw_texts):
            if check_url(text):
                continue
            if check_len(text):
                continue
            if check_accents(text):
                continue
            if check_punc(text):
                continue
            if "/" in text:
                continue
            if "“" in text:
                continue
            if '"' in text:
                continue
            if "”" in text:
                continue
            clean.write("\n")
            clean.write(text.strip())


def check_roman_number(word: str) -> bool:
    """
    Check a string if it probably is a roman number
    Args:
        word: (`str`) - word need to be checked

    Returns: ('bool') - True if word probably is a roman number.

    """
    for char in word:
        if char not in "xivXIVl":  # Sometime user use 'l' instead 'I'
            return False
    return True


def detect_notations(text: str) -> bool:
    """
    Detect notation in text.
    Args:
        text: A text string

    Returns: True if notation in text.

    """
    return any(notation in text for notation in NOTATIONS)


def detect_formulas(text: str) -> bool:
    """
    Detect formulas in text.
    Args:
        text(`str`): A text string

    Returns: True if formula in text.

    """
    # any word contain at least one number with any letter
    # or word with more than one uppercase letter
    # or word include special characters like (),-
    regex_object = re.compile(
        "([A-Za-z]+[\d@]+[\w@]*|[\d@]+[A-Za-z]+[\w@]*|\w*[A-Z]\w*[A-Z]\w*|\S+[(),-]{1,}\S+|cooh)"
    )
    formulas = regex_object.findall(text)
    return bool(formulas)


def find_repeat_char(text: str, repeat_limit: int) -> List:
    """

    Args:
        text (`str`): A text string
        repeat_limit (`int`): Minimum repeat times of a char to find

    Returns:
        List of repeat character in text
    """
    # The actual regex_expression is "(\w)\1{repeat_limit,}'}"
    # Which is help to find word character that repeat 'repeat_limit' times
    # We have to turn it to string to make sure the curly braces is string but not a word boundary
    regex_compiler = re.compile(rf"(\w)\1{'{' + str(repeat_limit) + ',}'}")
    repeat_char = [x.group() for x in regex_compiler.finditer(text)]
    return repeat_char


def is_special_token(word: str) -> bool:
    return word in ["<error>", "</error>", "$exception$"]


def find_word_with_repeat_char(
    text: str, repeat_limit: int
) -> List[Dict[str, Union[str, int, List[str]]]]:
    text = text.split(" ")
    word_with_repeat_char = [
        {"word": word, "word_index": idx, "repeat_chars": find_repeat_char(word, repeat_limit)}
        for idx, word in enumerate(text)
        if not is_special_token(word) and find_repeat_char(word, repeat_limit)
    ]
    return word_with_repeat_char


def remove_repeat_char(text: str, keep_duplicate_vowels: bool = True) -> str:
    """
    Remove duplicate characters in a sentence. Except roman number and formulas.
    Args:
        text: (`str`) A text need to be processed.

    Returns: (`str`) A text after processed

    """
    word_list = text.split()
    result = copy(word_list)

    if keep_duplicate_vowels:
        repeat_limit = 2
    else:
        repeat_limit = 1

    # Remove duplicate character (which is not vowels) in word (which is not roman number, formulas and special token)
    for i, word in enumerate(word_list):
        if (
            not check_roman_number(word)
            and not detect_formulas(word)
            and not is_special_token(word)
        ):
            for char in list(WORD_CHAR):
                if char not in VOWELS:
                    result[i] = re.sub(char + "+", char, result[i])

    # Remove remain duplicate character in word (which is not roman number, formulas and special token)
    # with repeat limit
    word_with_repeat_char = find_word_with_repeat_char(" ".join(result), repeat_limit=repeat_limit)
    while word_with_repeat_char:
        for item in word_with_repeat_char:
            word = item.get("word")
            repeat_chars = item.get("repeat_chars")
            word_index = item.get("word_index")
            if not detect_formulas(word) and not check_roman_number(word):
                for repeat_char in repeat_chars:
                    result[word_index] = result[word_index].replace(
                        repeat_char, repeat_char[:repeat_limit]
                    )
            del word_with_repeat_char[0]
    text = " ".join(result)
    return text


def remove_exception_error_tag(text: str) -> str:
    """
    Remove error tag for some exception (formulas, roman number, etc).
    Args:
        text (`str`): A text string

    Returns:
        Text string which was removed error tag for some exception.
    """
    temp_text = text.split()
    for idx, word in enumerate(temp_text):
        if not is_special_token(word) and (detect_formulas(word) or check_roman_number(word)):
            if temp_text[idx - 1] == "<error>":
                temp_text[idx - 1] = "$exception$"
            if temp_text[idx + 1] == "</error>":
                temp_text[idx + 1] = "$exception$"
    text = " ".join(temp_text).replace("$exception$", "")
    return remove_duplicate_space(text)


def normalize_detection(detection_list: List[str]) -> List[str]:
    """
    Remove <pad>, </s> in detection output
    Args:
        detection_list: List of strings need to be processed.

    Returns: List of strings after removed special token.

    """
    result = []
    for detection in detection_list:
        detection = detection.replace("<pad>", "")
        detection = detection.replace("</s>", "")
        detection = detection.strip()
        result.append(detection)
    return result


def remove_punc(text: str) -> str:
    """
    Remove punctuations in a string
    Args:
        text: (`str`) - text need to be removed punctuation

    Returns: (`str`) - text after remove punctuation

    """
    puncs = string.punctuation.replace("<", "").replace(">", "").replace("/", "")
    text = text.translate(str.maketrans(" ", " ", puncs))
    text = remove_duplicate_space(text)
    return text


def remove_duplicate_space(text: str) -> str:
    """
    Remove duplicate space characters in a string
    Args:
        text: (`str`) - String need to be cleaned

    Returns: (`str`) - String after cleaned

    """
    return re.sub(r"\s+", " ", text)


def clean_input_text(text: str) -> str:
    """
    Remove "\n" and space in text
    Args:
        text: (`str`) - Text need to be cleaned

    Returns: (`str`) - Text after removed "\n" and space character

    """
    text = text.replace("\n", "")
    text = text.strip()
    return text


def clean_error_tag(text_list: List[str]) -> List[str]:
    """
    Remove <error> token
    Args:
        text_list: List of text need to be cleaned

    Returns: List of text after cleaned

    """
    for text in text_list:
        yield remove_duplicate_space(text.replace("<error>", "").replace("</error>", "")).strip()
