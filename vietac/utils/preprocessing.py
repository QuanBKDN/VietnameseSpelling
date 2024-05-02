import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from vietac.dataset.clean import detect_formulas, detect_notations, remove_punc, remove_repeat_char
from vietac.utils.constants import MULTIPLE_ACRONYM, SINGLE_ACRONYM


def replace_acronym(text: str, non_single_acronym: bool = False) -> str:
    """
    Replace acronym by full explication for raw lowercase text.

    Args:
        text: a text string need to replace acronym.
        non_single_acronym: True if process non-single acronym.

    Returns:
        Text string was replaced acronym.

    """
    if non_single_acronym:
        for acronym, explication in MULTIPLE_ACRONYM.items():
            text = text.replace(acronym, explication)
        return text
    else:
        text = text.split(" ")
        for acronym, explication in SINGLE_ACRONYM.items():
            text = [explication if acronym == word else word for word in text]
        return " ".join(text)


def get_corrected_acronym(
    origin_text: list,
) -> List[Optional[Dict]]:  # , acronym: str, explication: str):
    """
    Get corrected words which is an acronym.
    Args:
        origin_text: List words of origin text.

    Returns:
        List of corrected words which is an acronym.

    """
    corrected_acronyms = []
    for acronym, explication in SINGLE_ACRONYM.items():
        corrected_acronym = [
            {
                "org_word": word,
                "corrected": explication,
                "org_word_position": index,
                "next_org_word_position": index + len(explication.split(" ")),
            }
            for index, word in enumerate(origin_text)
            if acronym == word
        ]
        corrected_acronyms.extend(corrected_acronym)
    return corrected_acronyms


def _count_multiple_acronym(text: str, multiple_acronym: str) -> int:
    """Count multiple acronym in text"""
    return text.count(multiple_acronym)


def _count_single_acronym(text: str, single_acronym: str) -> int:
    """Count single acronym in text"""
    text = text.split(" ")
    number_of_single_acronym = 0
    for word in text:
        if word == single_acronym:
            number_of_single_acronym += 1
    return number_of_single_acronym


def _add_error_tag_to_explication(
    text: str, explication: str, number_of_acronym: int, acronym: str
) -> str:
    """Add error tag into explication"""
    return text.replace(explication, "<error> " + acronym + " </error>", number_of_acronym)


def add_error_tag_into_explication_for_detected_error_text(
    origin_text: str, detected_error_text: str
) -> str:
    """
    Add error tag to explication for detected error text.
    Args:
        origin_text: Text that haven't preprocessing.
        detected_error_text: Text that was detected error.

    Returns:
        Text added error tag into explication.

    """
    for acronym, explication in MULTIPLE_ACRONYM.items():
        number_of_acronym = _count_multiple_acronym(text=origin_text, multiple_acronym=acronym)
        detected_error_text = _add_error_tag_to_explication(
            text=detected_error_text,
            explication=explication,
            number_of_acronym=number_of_acronym,
            acronym=acronym,
        )
    for acronym, explication in SINGLE_ACRONYM.items():
        number_of_acronym = _count_single_acronym(text=origin_text, single_acronym=acronym)
        detected_error_text = _add_error_tag_to_explication(
            text=detected_error_text,
            explication=explication,
            number_of_acronym=number_of_acronym,
            acronym=acronym,
        )
    return detected_error_text


def _count_error(text: str) -> int:
    return text.count("<error>")


def calculating_error_rate_in_text(origin_text: str, detected_error_text: str) -> float:
    """
    Calculating error rate in text that was detected error.
    Args:
        origin_text: Text that haven't preprocessing.
        detected_error_text: Text that was detected error.

    Returns:
        Error rate in text.
    """
    number_of_error = _count_error(detected_error_text)
    number_of_words = len(origin_text.split(" "))
    return number_of_error / number_of_words


def pre_correct(inputs_list: List) -> List:
    """
    Correct simple errors by dictionary
    Args:
        inputs_list: List of text need to be correct

    Returns: List of text after corrected

    """
    result = []
    for text in inputs_list:
        text = remove_punc(text)
        text = replace_acronym(text)
        text = remove_repeat_char(text)
        result.append(text)
    return result


def _is_there_continue_error(text: str) -> bool:
    """Check if there is continued error in text."""
    reg_continue_error = "</error> <error>"
    continue_error = re.search(reg_continue_error, text, re.IGNORECASE)
    return bool(continue_error)


def _is_start_with_error(text: str) -> bool:
    """Check if text begin with an error."""
    reg_start_with_error = "<error>"
    return text.startswith(reg_start_with_error)


def _is_end_with_error(text: str) -> bool:
    """Check if text end with an error."""
    reg_end_with_error = "</error>"
    return text.endswith(reg_end_with_error)


def _is_there_error(text: str) -> bool:
    """Check if there is error token in text."""
    reg_error = "<error>"
    return reg_error in text


def _remove_continue_error_tag(text: str) -> str:
    """Remove continue error tag."""
    reg_continue_error = " </error> <error> "
    return text.replace(reg_continue_error, " ")


def get_list_of_error_index(text: str) -> List[Dict]:
    error_index_list = []
    regex_object = re.compile("(?<=<error>)(.*?)(?=</error>)")
    num_error = 0
    for match in regex_object.finditer(text):
        error_token_length = 8 + num_error * 17
        index_of_the_starting_position_of_the_match = match.start()
        length_of_the_match = len(match.group().strip())

        start = index_of_the_starting_position_of_the_match - error_token_length + 1
        end = start + length_of_the_match

        error_index_list.append({"start": start, "end": end})
        num_error += 1
    return error_index_list


def get_error_index(list_of_error_index: List[Dict], index: int) -> Optional[Dict]:
    try:
        error = list_of_error_index[index]
    except IndexError:
        warnings.warn(f"Length of {list_of_error_index} is shorter than {index}.")
        error = None
    return error


def extract_clean_starting_text(text: str) -> List[str]:
    """Return starting text that have no error."""
    reg_clean_starting_text = "(.*?) <error>"
    clean_starting_text = re.search(reg_clean_starting_text, text, re.IGNORECASE)
    if clean_starting_text:
        return clean_starting_text.group(1).split(" ")
    else:
        warnings.warn("Can not extract clean starting text.")
    return list()


def extract_clean_ending_text(text: str) -> List[str]:
    """Return ending text that have no error."""
    reg_clean_ending_text = "</error> "
    clean_ending_text = text.rsplit(reg_clean_ending_text, 1)[1].strip().split(" ")
    if clean_ending_text:
        return clean_ending_text
    else:
        warnings.warn("Can not extract clean ending text.")
    return list()


def extract_clean_text(text: str) -> Tuple[Union[List[str], list], Union[List[str], list]]:
    """Return starting and ending text that have no error."""
    return check_and_extract_clean_starting_text(text), check_and_extract_clean_ending_text(text)


def extract_error_words(text: str) -> List[str]:
    """Return word that predicted as an error."""
    re_error_words = "<error> (.*?) </error>"
    error_words = re.findall(re_error_words, text, re.IGNORECASE)
    if not len(error_words):
        warnings.warn("This sentence does not contain grammar error.")
    return error_words


def extract_non_error_words(text: str) -> List[str]:
    """Return word is not predicted as an error, and not in clean starting or clean ending text."""
    re_non_error_words = "</error>?(.*?) <error>"
    non_error_words = re.findall(re_non_error_words, text, re.IGNORECASE)
    if len(non_error_words) == 0:
        warnings.warn("This sentence have no correct grammar.")
        return non_error_words
    non_error_words = " ".join(word.strip() for word in non_error_words)
    return non_error_words.split(" ")


def extract_remain_text_to_arrange(
    corrected_text: List[str],
    clean_starting_text: Union[List[str], list],
    clean_ending_text: Union[list, List[str]],
) -> List[str]:
    """Return remain text have to arrange (start with an error word)."""
    return corrected_text[
        len(clean_starting_text) : (len(corrected_text) - len(clean_ending_text))
    ]


def check_and_extract_clean_starting_text(text: str) -> Union[list, List[str]]:
    """Return starting text have no error."""
    if _is_start_with_error(text):
        return list()
    return extract_clean_starting_text(text)


def check_and_extract_clean_ending_text(text: str) -> Union[list, List[str]]:
    """Return ending text have no error."""
    if _is_end_with_error(text):
        return list()
    return extract_clean_ending_text(text)


def add_predicted_and_origin_to_dict(
    origin_word: str,
    corrected_words: List[str],
    index_origin_word: Optional[Dict[str, Any]] = None,
):
    dict_of_predicted_and_origin = {"origin": origin_word, "corrected": corrected_words}
    if index_origin_word is None:
        return dict_of_predicted_and_origin
    dict_of_predicted_and_origin.update(index_origin_word)
    return dict_of_predicted_and_origin


def _arrange_corrected_to_origin_algorithm(
    arranged_text: List[Dict],
    remain_text_to_arrange: List[str],
    non_error_words: List[str],
    error_words: List[str],
    predicted_error_text: str,
) -> List[Dict]:
    index_error_word = 0
    index_non_error_word = 0
    explication = []
    error_index = get_list_of_error_index(predicted_error_text)

    for corrected_word in remain_text_to_arrange:
        if not non_error_words:
            arranged_text.append(
                add_predicted_and_origin_to_dict(
                    origin_word=" ".join(error_words),
                    corrected_words=remain_text_to_arrange,
                    index_origin_word=error_index[index_error_word],
                )
            )
            break
        elif corrected_word not in non_error_words[index_non_error_word]:
            explication.append(corrected_word)
        else:
            if explication:
                arranged_text.append(
                    add_predicted_and_origin_to_dict(
                        origin_word=error_words[index_error_word],
                        corrected_words=explication,
                        index_origin_word=error_index[index_error_word],
                    )
                )
                explication = []
                index_error_word += 1

            if index_non_error_word == len(non_error_words) - 1:
                arranged_text.append(
                    add_predicted_and_origin_to_dict(
                        origin_word=error_words[index_error_word],
                        corrected_words=explication,
                        index_origin_word=error_index[index_error_word],
                    )
                )
                continue
            else:
                index_non_error_word += 1
    for item in arranged_text:
        item.update({"corrected": " ".join(item["corrected"])})
    return arranged_text


def _arrange_corrected_to_origin_for_no_error_corrected(
    list_of_predicted_error_text: List[str], list_of_corrected_text: List[str], arranged_text: List
) -> List:
    """Arrange text that have no error detected."""
    for origin_word, corrected_word in zip(list_of_predicted_error_text, list_of_corrected_text):
        arranged_text.append(add_predicted_and_origin_to_dict(origin_word, [corrected_word]))
    return arranged_text


def arrange_corrected_to_origin(corrected_text: str, predicted_error_text: str) -> List[Dict]:
    """
    Arrange/mapping corrected word in from model output text to the origin/predicted error text.

    Args:
        corrected_text: Text that was corrected by model.
        predicted_error_text: Text that was predicted error. Having or not having error tokens.

    Returns:
        A list contain multiple dictionary, each dictionary have origin word as key and predict words as value.
        For example: [{'This':'This'}, {'is':'is'}, {'origin':'corrected'}, , {'text':'text'}]
    """
    arranged_text = []
    list_of_corrected_text = corrected_text.split(" ")
    list_of_predicted_error_text = predicted_error_text.split(" ")

    if not _is_there_error(predicted_error_text) or len(list_of_predicted_error_text) == len(
        list_of_corrected_text
    ):
        return arranged_text

    else:
        if _is_there_continue_error(predicted_error_text):
            predicted_error_text = _remove_continue_error_tag(predicted_error_text)

        clean_starting_text, clean_ending_text = extract_clean_text(predicted_error_text)

        non_error_words = extract_non_error_words(predicted_error_text)
        error_words = extract_error_words(predicted_error_text)

        remain_text_to_arrange = extract_remain_text_to_arrange(
            list_of_corrected_text, clean_starting_text, clean_ending_text
        )

        return _arrange_corrected_to_origin_algorithm(
            arranged_text=arranged_text,
            remain_text_to_arrange=remain_text_to_arrange,
            non_error_words=non_error_words,
            error_words=error_words,
            predicted_error_text=predicted_error_text,
        )


def extract_unprocessed_sample(
    text_list: List[str], detectors: Optional[List[Callable]] = None
) -> Tuple[List[str], List[int]]:
    """
    Extract exception from input list
    Args:
        detectors : List func to detect unprocessed cases
        text_list: Input list

    Returns: A tuple contains (Processed list, Unprocessed index)

    """
    if detectors is None:
        detectors = [detect_formulas, detect_notations]
    unprocessed_index = []
    processed_list = []
    for i, text in enumerate(text_list):
        if any([detector(text) for detector in detectors]) or text == "":
            unprocessed_index.append(i)
        else:
            processed_list.append(text)

    return processed_list, unprocessed_index
