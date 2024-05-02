import re
from copy import copy
from math import ceil
from random import choice, randint, random, sample
from typing import Callable, List, Tuple, Union

import numpy as np
from numpy import ndarray
from vietac.utils.constants import (
    ACCENT_MAP,
    ALPHABET,
    DIACRITICS_MAP,
    HEAD_NORMAL_ERROR_MAP,
    TAIL_NORMAL_ERROR_MAP,
    TELEX_ACCENT_MAP,
    TELEX_DIACRITICS_MAP,
    add_accents_map,
    common_error_map,
    misspell_head_consonant_map,
    misspell_tail_consonant_map,
    misspell_vowel_map,
    remove_accent_map,
    remove_diacritics_map,
    telex_dup_map,
    telex_map,
)


def _create_list_index_of_error_word(
    start_index_of_error_words: int,
    number_of_error_words: int,
    end_index_of_error_words: int,
    index_of_error_words: List[int],
):
    """
    Create list index of word need to create error following ratio of error type.

    Args:
        start_index_of_error_words: Start index of list of error words.
        number_of_error_words: Total number of error words.
        end_index_of_error_words: End index of list of error words.
        index_of_error_words:  List index of error words.

    Returns:
        List index of word need to create error following ratio of error type.

    """
    if start_index_of_error_words + 1 > number_of_error_words:
        return None  # cannot add more error word into sentence
    elif end_index_of_error_words > number_of_error_words:
        return index_of_error_words[start_index_of_error_words:]
    return index_of_error_words[start_index_of_error_words:end_index_of_error_words]


def _add_error_into_sentence(
    error_word_indexes: List[int], error_sentence: List[str], function_create_error: Callable,
):
    """Create error word with given function into sentence"""
    for index_of_error_word in error_word_indexes:
        error_word, is_create_error = function_create_error(error_sentence[index_of_error_word])
        error_sentence[index_of_error_word] = error_word  # replace origin word by error word
    return error_sentence


def create_error_with_control(
    sentence: str,
    error_ratio: Union[float, int],
    functions_create_error: List[Callable],
    ratios_error_type: List[Union[float, int]],
):
    """
    An error controller for creating error sentence. If the sentence length is not enough for all required error type,
    the error type appear in sentence will be selected proportionally according to position in list of error type,
    while the error ratio still kept.

    Args:
        sentence: Sentence need to be created error.
        error_ratio: Ratio of error that the sentence will have. If > 1.0, the error ratio will be downgraded to 1.0
                    automatically
        functions_create_error: List functions for error type will be used to create error.
        ratios_error_type: List ratio of error type, each ratio map to error type according to position in list.

    Returns:
        Error sentence with given error ratio.

    """
    error_sentence = sentence.split(" ")
    number_of_word_in_sentence = len(error_sentence)
    number_of_error_words = ceil(error_ratio * number_of_word_in_sentence)
    index_of_error_words = sample(range(number_of_word_in_sentence), number_of_error_words)
    if error_ratio > 1:
        error_ratio = 1
    number_of_error_word = [
        ceil(error_ratio * number_of_error_words) for error_ratio in ratios_error_type
    ]

    start_index_of_error_words = 0
    for index_of_function, function_create_error in enumerate(functions_create_error):
        end_index_of_error_words = (
            start_index_of_error_words + number_of_error_word[index_of_function]
        )

        error_word_indexes = _create_list_index_of_error_word(
            start_index_of_error_words,
            number_of_error_words,
            end_index_of_error_words,
            index_of_error_words,
        )
        if error_word_indexes is not None:
            error_sentence = _add_error_into_sentence(
                error_word_indexes, error_sentence, function_create_error
            )
            start_index_of_error_words = end_index_of_error_words
        else:
            break

    return " ".join(error_sentence)


def remove_vn_special(sentence: str) -> str:
    """
    Remove Vietnamese special characters in a sentence
    Args:
        sentence: (`str`) sentence need to be processed

    Returns: (`str`) sentence after removed Vietnamese special characters

    """
    word_list = sentence.split(" ")
    error_sentence = []
    for word in word_list:
        aug_word = [*word]
        for i, char in enumerate(word):
            if char == "đ":
                aug_word[i] = "d"
            else:
                for accent_key, accent_value in ACCENT_MAP.items():
                    if char in accent_value:
                        for diacritics_key, diacritics_value in DIACRITICS_MAP.items():
                            if accent_key in diacritics_value:
                                aug_word[i] = diacritics_key
        error_sentence.append("".join(aug_word))
    return " ".join(error_sentence)


def add_normal_char_error(
    sentence: str, error_rate: Union[List[float], ndarray], ratio: float = 0.1
) -> Tuple[str, List[float]]:
    """
    Add some common errors to a sentence
    Args:
        sentence: (`str`) Sentence need to be processed.
        error_rate: (`List[float]`) Error rates for each word
        ratio: (`float`) Error ratio

    Returns: (`Tuple[str, List[float]]`)

    """
    word_list = sentence.split(" ")
    error_sentence = []
    for i, word in enumerate(word_list):
        aug_word = [*word]
        for key, value in HEAD_NORMAL_ERROR_MAP.items():
            if aug_word[0] == key:
                word_error_rate = random()
                if word_error_rate < ratio:
                    aug_word[0] = HEAD_NORMAL_ERROR_MAP[key]
                    error_rate[i] = 0
                break
        for key, value in TAIL_NORMAL_ERROR_MAP.items():
            if aug_word[-1] == key:
                word_error_rate = random()
                if word_error_rate < ratio:
                    aug_word[-1] = TAIL_NORMAL_ERROR_MAP[key]
                    error_rate[i] = 0
                break
        error_sentence.append("".join(aug_word))
    return " ".join(error_sentence), error_rate


def add_telex_accent_error(sentence: str, error_rate: List[float], ratio: float = 0.1):
    """
    Add telex typing errors to a sentence
    Args:
        sentence: (`str`) Sentence need to be processed.
        error_rate: (`List[float]`) Error rates for each word
        ratio: (`float`) Error ratio

    Returns: (`Tuple[str, List[float]]`)

    """
    word_list = sentence.split(" ")
    error_sentence = []
    for index, word in enumerate(word_list):
        aug_word = [*word]
        for i, char in enumerate(word):
            for telex_key, accent in TELEX_ACCENT_MAP.items():
                if char in accent:
                    rate = ratio * error_rate[index]
                    word_error_rate = random()
                    if word_error_rate < rate:
                        for key, value in ACCENT_MAP.items():
                            if char in value:
                                aug_word[i] = key
                                aug_word.append(telex_key)
                                error_rate[index] = error_rate[index] * 0.7
            for telex_key, diacritics in TELEX_DIACRITICS_MAP.items():
                if char == diacritics:
                    word_error_rate = random()
                    if word_error_rate < ratio:
                        aug_word[i] = telex_key
                        error_rate[index] = error_rate[index] * 0.7
        error_sentence.append("".join(aug_word))
    return " ".join(error_sentence), error_rate


def add_diacritics_error(
    sentence: str, error_rate: List[float], ratio: float = 0.1
) -> Tuple[str, List[float]]:
    """
    Add diacritics typing errors to a sentence
    Args:
        sentence: (`str`) Sentence need to be processed.
        error_rate: (`List[float]`) Error rates for each word
        ratio: (`float`) Error ratio

    Returns: (`Tuple[str, List[float]]`)

    """
    word_list = sentence.split(" ")
    error_sentence = []
    for idx, word in enumerate(word_list):
        aug_word = [*word]
        word_error_rate = random()
        rate = ratio * error_rate[idx]
        if word_error_rate < rate:
            for i, char in enumerate(word):
                for key, value in DIACRITICS_MAP.items():
                    if char in value:
                        error_char = char
                        while error_char == char:
                            error_accent_idx = randint(0, len(value) - 1)
                            error_char = value[error_accent_idx]
                            aug_word[i] = value[error_accent_idx]
                            error_rate[idx] = error_rate[idx] * 0.7
                double_error = random()
                if double_error > 0.2:
                    break
        error_sentence.append("".join(aug_word))
    return " ".join(error_sentence), error_rate


def add_accent_error(
    sentence: str, error_rate: List[float], ratio: float = 0.1
) -> Tuple[str, List[float]]:
    """
    Add accent errors to a sentence
    Args:
        sentence: (`str`) Sentence need to be processed.
        error_rate: (`List[float]`) Error rates for each word
        ratio: (`float`) Error ratio

    Returns: (`Tuple[str, List[float]]`)

    """
    word_list = sentence.split(" ")
    error_sentence = []
    for idx, word in enumerate(word_list):
        aug_word = [*word]
        rate = ratio * error_rate[idx]
        word_error_rate = random()
        if word_error_rate < rate:
            for i, char in enumerate(word):
                for key, value in ACCENT_MAP.items():
                    if char in value:
                        error_accent_idx = randint(0, 5)
                        aug_word[i] = value[error_accent_idx]
                        error_rate[idx] = error_rate[idx] * 0.7
                double_error = random()
                if double_error > 0.2:
                    break
        error_sentence.append("".join(aug_word))
    return " ".join(error_sentence), error_rate


def add_characters_error(
    sentence: str, error_rate: List[float], ratio: float = 0.05
) -> Tuple[str, List[float]]:
    """
    Add missing/redundancy character errors to a sentence
    Args:
        sentence: (`str`) Sentence need to be processed.
        error_rate: (`List[float]`) Error rates for each word
        ratio: (`float`) Error ratio

    Returns: (`Tuple[str, List[float]]`)

    """
    word_list = sentence.split(" ")
    error_sentence = []
    for idx, word in enumerate(word_list):
        aug_word = [*word]
        if len(word) > 1:
            word_error_rate = random()
            rate = ratio * error_rate[idx]
            if word_error_rate < rate:
                change = True
                while change:
                    i = randint(0, len(word) - 1)
                    if random() < 0.5:
                        error_id = randint(0, 2)
                        if error_id == 0:
                            char_id = randint(0, 27)
                            aug_word[i] = ALPHABET[char_id]
                        elif error_id == 1 and len("".join(aug_word)) > 1:
                            aug_word[i] = ""
                            break
                        else:
                            char_id = randint(0, 27)
                            aug_word.insert(i + 1, ALPHABET[char_id])
                            error_rate[idx] = error_rate[idx] * 0.7
                        double_error = random()
                        if double_error > 0.2:
                            change = False
        error_sentence.append("".join(aug_word))
    return " ".join(error_sentence), error_rate


def augment_error(sentence: str) -> str:
    """
    Combine all error functions
    Args:
        sentence: (`str`) Sentence need to be augmented

    Returns: Sentence after augmented

    """
    error_rate = np.ones(len(sentence.split(" ")))
    remove_all_accents = random()
    out, error_rate = add_normal_char_error(sentence, error_rate)

    if remove_all_accents < 0.06:
        out = remove_vn_special(sentence)
    else:
        out, error_rate = add_telex_accent_error(sentence, error_rate)
        out, error_rate = add_accent_error(out, error_rate)
        out, error_rate = add_diacritics_error(out, error_rate)
        out, error_rate = add_characters_error(out, error_rate)
    return out


def add_error_tag(original_sentence: str, error_sentence: str) -> Tuple[str, str]:
    """
    Add <error> </error> tag into error sentence.
    Args:
        original_sentence: (`str`) The original sentence
        error_sentence: (`str`) The error sentence

    Returns: (`Tuple[str, str]`) Error sentence with <error> tag

    """
    tagged_sentence = error_sentence.split()
    tagged_only_corrected = original_sentence.split()
    original_sentence = original_sentence.split()
    correct_target = original_sentence.copy()
    error_sentence = error_sentence.split()
    classify_target = []
    num_error = 0
    new_target = ""
    for i in range(len(original_sentence)):
        try:
            if original_sentence[i] != error_sentence[i]:
                tagged_sentence.insert(num_error * 2 + i, "<error>")
                tagged_sentence.insert(num_error * 2 + i + 2, "</error>")

                correct_target.insert(num_error * 2 + i, "<correct>")
                correct_target.insert(num_error * 2 + i + 2, "</correct>")

                tagged_only_corrected.insert(num_error * 2 + i, "<corrected>")
                tagged_only_corrected.insert(num_error * 2 + i + 2, "</corrected>")
                classify_target.append(0)
                num_error += 1
            else:
                classify_target.append(1)
            new_target = " ".join(correct_target)
            corrected = re.findall(r"(?<=\<correct> )(.*?)(?=\ <\/correct>)", new_target)
            if corrected:
                new_target = " <corrected> ".join(corrected)
            else:
                new_target = "<non_error>"
        except IndexError:
            raise IndexError
    return " ".join(tagged_sentence), new_target


def remove_diacritics(word: str) -> Tuple[str, int]:
    """
    Remove diacritics in a word.
    Args:
        word: (`str`) - Word need to be Remove diacritics.
    Returns: (Tuple[str, int]) - Word after removed diacritics and number of errors created.

    """
    char_list = [*word]
    has_diacritic_chars = {}
    for idx, char in enumerate(char_list):
        if char in remove_diacritics_map.keys():
            has_diacritic_chars[char] = idx
    num_errors = 0
    if has_diacritic_chars:
        while num_errors == 0:
            for char in has_diacritic_chars:
                is_error = 1 if random() < 0.4 else 0
                if is_error:
                    num_errors += 1
                    char_list[has_diacritic_chars[char]] = remove_diacritics_map[char]
    return "".join(char_list), num_errors


def remove_accents(word: str) -> Tuple[str, int]:
    """
    Remove accents in a word.
    Args:
        word: (`str`) - Word need to be Remove accents.
    Returns: (Tuple[str, int]) - Word after removed accents and number of errors created.

    """
    char_list = [*word]
    has_accent_chars = {}
    for idx, char in enumerate(char_list):
        if char in remove_accent_map.keys():
            has_accent_chars[char] = idx
    num_errors = 0
    if has_accent_chars:
        while num_errors == 0:
            for char in has_accent_chars:
                is_error = 1 if random() < 1 else 0
                if is_error:
                    num_errors += 1
                    char_list[has_accent_chars[char]] = remove_accent_map[char]
    return "".join(char_list), num_errors


def add_accents(word: str) -> Tuple[str, int]:
    """
    Add accents in a word.
    Args:
        word: (`str`) - Word need to be Add accents.
    Returns: (Tuple[str, int]) - Word after Added accents and number of errors created.

    """
    char_list = [*word]
    vowel_char = {}
    for idx, char in enumerate(char_list):
        if char in add_accents_map.keys():
            vowel_char[char] = idx
    num_errors = 0
    if vowel_char:
        num_errors = 0
        while num_errors == 0:
            for char in vowel_char:
                char_list[vowel_char[char]] = choice(add_accents_map[char])
                num_errors += 1

    return "".join(char_list), num_errors


def delete_char(word: str, n: int = 1) -> Tuple[str, int]:
    """
    Delete random characters in a word
    Args:
        word: (`str`) - Word need to be alter.
        n: (`int`) - Number of random characters will be delected.

    Returns: (Tuple[`str`,`int`]) - Word after altered and number of errors created.

    """
    if len(word) <= n:
        return word, 0
    char_list = [*word]
    num_errors = 0
    while num_errors < n:
        del_index = randint(0, len(word) - 1)
        del char_list[del_index]
        word = "".join(char_list)
        num_errors += 1
    return word, num_errors


def insert_char(word: str, n: int = 1) -> Tuple[str, int]:
    """
    Insert random characters into a word
    Args:
        word: (`str`) - Word need to be insert.
        n: (`int`) - Number of random characters will be inserted.

    Returns: (Tuple[`str`,`int`]) - Word after Insert and number of errors created.

    """
    char_list = [*word]
    num_errors = 0
    while num_errors < n:
        insert_index = randint(0, len(word) - 1)
        char_insert = choice(ALPHABET)
        char_list[insert_index] = char_list[insert_index] + char_insert
        word = "".join(char_list)
        num_errors += 1
    return word, num_errors


def add_misspell_head_consonant(word: str) -> Tuple[str, int]:
    """
    Add head consonant misspell errors into a word
    Args:
        word: (`str`) - Word need to be alter.

    Returns: (Tuple[`str`,`int`]) - Word after altered and number of errors created.

    """
    if word[:2] in misspell_head_consonant_map:
        word = word.replace(word[:2], choice(misspell_head_consonant_map[word[:2]]))
        return word, 1
    char_list = [*word]
    if word[0] in misspell_head_consonant_map:
        char_list[0] = choice(misspell_head_consonant_map[char_list[0]])
        return "".join(char_list), 1
    return "".join(char_list), 0


def add_misspell_tail_consonant(word: str) -> Tuple[str, int]:
    """
    Add tail consonant misspell errors into a word
    Args:
        word: (`str`) - Word need to be alter.

    Returns: (Tuple[`str`,`int`]) - Word after altered and number of errors created.

    """
    if word[-3:] in misspell_tail_consonant_map:
        char_list = [*word]
        char_list[-3:] = ["", "", ""]
        char_list.append(choice(misspell_tail_consonant_map[word[-3:]]))
        return "".join(char_list), 1
    if word[-2:] in misspell_tail_consonant_map:
        char_list = [*word]
        char_list[-2:] = ["", ""]
        char_list.append(choice(misspell_tail_consonant_map[word[-2:]]))
        return "".join(char_list), 1
    if word[-1:] in misspell_tail_consonant_map:
        char_list = [*word]
        char_list[-1:] = [""]
        char_list.append(choice(misspell_tail_consonant_map[word[-1:]]))
        return "".join(char_list), 1

    char_list = [*word]
    if word[0] in misspell_tail_consonant_map:
        char_list[0] = choice(misspell_tail_consonant_map[char_list[0]])
        return "".join(char_list), 1
    return "".join(char_list), 0


def add_misspell_vowel(word: str) -> Tuple[str, int]:
    """
    Add vowel misspell errors into a word
    Args:
        word: (`str`) - Word need to be alter.

    Returns: (Tuple[`str`,`int`]) - Word after altered and number of errors created.

    """
    for key in misspell_vowel_map:
        if key in word:
            return word.replace(key, misspell_vowel_map[key]), 1
    return word, 0


def swap_special_char(word: str) -> List[str]:
    """
    Create all telex typing cases from a word
    Args:
        word: (`str`) - word has Vietnamese special characters ("s", "x", "r", "j", "f", "w")
    Returns: (List[str]) - Return a list contains all telex typing cases

    """
    accents = ["s", "x", "r", "j", "f"]
    results = []
    regex = r"(.)\1+"
    matches = re.finditer(regex, word)
    telex_group = ""
    if word.count("w") == 1:

        results.append(word.replace("w", "") + "w")
        if word[-1] in accents:
            accent = word[-1]
            results.append(word.replace("w", "")[:-1] + "w" + accent)
    try:
        for match in matches:
            telex_group = match.group()
            break
        if word[-1] in accents:
            accent = word[-1]
            results.append(
                word.replace(telex_group, telex_group[0])[:-1] + telex_group[0] + accent
            )
        results.append(word.replace(telex_group, telex_group[0]) + telex_group[0])
    except IndexError:
        return results
    return results


def telex_transform_base(word: str) -> Tuple[str, str]:
    copy_word = copy(word)
    for telex in telex_dup_map:
        if telex in copy_word:
            copy_word = copy_word.replace(telex, telex_dup_map[telex])
    i = 0
    len_word = len(word)
    while i < len_word:
        if word[i] in telex_map:
            char_list = [*word]
            char_list[i] = telex_map[char_list[i]]
            word = "".join(char_list)
            len_word = len(word)
        i += 1
    return word, copy_word


def telex_transform(word: str) -> List[str]:
    """
    Transform a Vietnamese word into telex typing
    Args:
        word: Vietnamese word has telex special characters

    Returns: Return a list contains all telex typing cases

    """
    word, copy_word = telex_transform_base(word)
    words = []
    try:
        diacritic = re.findall(r"-\w", word)
        words.append(word.replace(diacritic[0], diacritic[0][-1]))
        word = word.replace(diacritic[0], "")
        word = word + diacritic[0][-1]
        words.append(word)

    except IndexError:
        words.append(word)
    try:
        diacritic = re.findall(r"-\w", copy_word)
        words.append(copy_word.replace(diacritic[0], diacritic[0][-1]))
        words.append(copy_word.replace(diacritic[0], "") + diacritic[0][-1])

    except IndexError:
        words.append(copy_word)
    results = []
    results.extend(words)
    for word in words:
        results.extend(swap_special_char(word))
    return list(set(results))


def add_telex_errors(word: str) -> Tuple[str, int]:
    """
    Add telex error into a word
    Args:
        word: (`str`) - Word need to be add errors.

    Returns: (Tuple[`str`,`int`]) - Error word and number of errors created.

    """
    error_list = telex_transform(word)
    if len(error_list) > 1:
        try:
            error_list.remove(word)
        except ValueError:
            pass
    telex_error = choice(error_list)
    if telex_error == word:
        return word, 0
    return telex_error, 1


class ErrorsWrapper:
    """
    Make a pipeline to apply multiple errors on single word.
    """

    def __init__(self, error_functions: List[Callable]):
        """
        Args:
            error_functions: List of error creation functions
        """
        self.error_functions = error_functions

    def __call__(self, *args, **kwargs):
        try:
            if "word" in kwargs:
                word = kwargs["word"]
            else:
                word = args[0]
        except IndexError:
            raise Exception("Pass a word by *args || **kwargs with key `word`")
        num_errors = 0
        for func in self.error_functions:
            word, n_e = func(word)
            num_errors += n_e
        return word, num_errors


def add_common_errors(sentence: str) -> str:
    """
    Add common errors (teen code) into a sentence
    Args:
        sentence: (`str`) - Sentence need to be added errors

    Returns: (`str`) - Sentence after added errors

    """
    result = []
    word_list = sentence.split()
    for word in word_list:
        if word in common_error_map:
            if random() < 0.35:
                result.append(common_error_map[word])
                continue
        result.append(word)
    return " ".join(result)
