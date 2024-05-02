from typing import Dict, List, Optional

from vietac.dataset.augment_error import telex_transform_base
from vietac.dataset.clean import normalize_detection, remove_duplicate_space
from vietac.utils.preprocessing import get_corrected_acronym


def matching(detection_list: List, correction_list: List, matching_threshold: float = 0.5) -> List:
    """
    Match predict word from model output into input sentence
    Args:
        detection_list: List of detected sentences
        correction_list: List of corrected
        matching_threshold: Reject prediction if matching rate less than this threshold

    Returns: Result sentence after mapped correct words to error words

    """
    # For corrected-only target
    detection = normalize_detection(detection_list)
    prediction = normalize_detection(correction_list)
    result = []
    for i in range(len(detection)):
        detect_sentence = detection[i]
        detect_sentence = remove_duplicate_space(
            detect_sentence.replace("<error>", "").replace("</error>", "")
        )
        detect_sentence = detect_sentence.split()
        predict_sentence = prediction[i].split()
        if len(detect_sentence) == len(predict_sentence):
            correct_sentence = []
            for idx, word in enumerate(detect_sentence):
                error_word = telex_transform_base(word)[0].replace("-", "")
                predict_word = telex_transform_base(predict_sentence[idx])[0].replace("-", "")
                matching_rate = calculating_character_matching_rate(error_word, predict_word)
                if matching_rate >= matching_threshold:
                    correct_sentence.append(predict_sentence[idx])
                else:
                    correct_sentence.append(word)
            result.append(" ".join(correct_sentence))
        else:
            result.append(" ".join(detect_sentence))
    return result


def intersection(lst1: str, lst2: str) -> List[str]:
    """
    Find intersect characters between 2 strings
    Args:
        lst1: First string
        lst2: Second string

    Returns: List of intersect characters
    """
    return list(set(lst1) & set(lst2))


def clean_suggestion(suggestion_list: List[dict]) -> List[dict]:
    """
    Remove unchange words
    Args:
        suggestion_list: List of word by word suggestions
    Returns: Cleaned suggestion list

    """
    i = 0
    while i < len(suggestion_list):
        if suggestion_list[i]["origin"] == suggestion_list[i]["corrected"]:
            suggestion_list.pop(i)
        else:
            i += 1
    return suggestion_list


def calculating_character_matching_rate(error_word: str, corrected_word: str) -> float:
    """
    Calculating match rate of characters between error word and corrected word
    Args:
        error_word: Word that was detected as error.
        corrected_word: Word that was corrected by model.

    Returns:
        Matching rate of characters between error word and corrected word

    """
    character_matching = len(intersection(error_word, corrected_word))
    return character_matching / len(corrected_word)


def detect_unknown_token(text: str) -> bool:
    """
    Detect unknown token in text.
    Args:
        text: A text string.

    Returns:
        True if unknown token in text.
    """
    return "<unk>" in text


def process_unknown_token(output_texts: List[str], origin_text: List[str]) -> List[str]:
    for idx, text in enumerate(output_texts):
        if detect_unknown_token(text):
            output_texts[idx] = origin_text[idx]
    return output_texts


def get_corrected_non_acronym(
    corrected_text: List[str], origin_text: List[str], corrected_acronyms: List[Optional[Dict]]
) -> List[Optional[Dict]]:
    """
    Get corrected words which is not an acronym.

    Args:
        corrected_text: List words of corrected text.
        origin_text: List words of origin text.
        corrected_acronyms: List of corrected word which is acronyms.

    Returns:
        List of corrected words which is not an acronym.
    """
    idx_corrected_word = 0
    corrected_words = []
    max_idx_corrected_word = len(corrected_text) - 1
    corrected_words.extend(corrected_acronyms)
    for idx, origin_word in enumerate(origin_text):
        if idx_corrected_word > max_idx_corrected_word:
            break
        else:
            is_acronym_word = [
                corrected_acronym["next_org_word_position"]
                for corrected_acronym in corrected_acronyms
                if corrected_acronym["org_word_position"] == idx
            ]
            if is_acronym_word:
                idx_corrected_word = is_acronym_word[0]
                continue
            if origin_word != corrected_text[idx_corrected_word]:
                corrected_words.append(
                    {
                        "org_word": origin_word,
                        "corrected": corrected_text[idx_corrected_word],
                        "org_word_position": idx,
                        "next_org_word_position": idx_corrected_word + 1,
                    }
                )
            idx_corrected_word += 1
    return corrected_words


def get_corrected_word(origin_text: str, corrected_text: str) -> List[Optional[Dict]]:
    """
    Get all corrected words which were arranged to origin word.
    Args:
        origin_text: Origin text string
        corrected_text: Corrected text string

    Returns:
        All corrected words which were arranged to origin word.
    """
    origin_text = " ".join(origin_text.strip().split()).split(" ")
    corrected_text = " ".join(corrected_text.strip().split()).split(" ")
    if len(origin_text) == len(corrected_text):
        corrected_words = [
            {
                "org_word": origin_word,
                "corrected": corrected_word,
                "org_word_position": index,
                "next_org_word_position": index + 1,
            }
            for index, (origin_word, corrected_word) in enumerate(zip(origin_text, corrected_text))
            if origin_word != corrected_word
        ]
    else:
        corrected_acronyms = get_corrected_acronym(origin_text)
        corrected_words = get_corrected_non_acronym(
            corrected_text=corrected_text,
            origin_text=origin_text,
            corrected_acronyms=corrected_acronyms,
        )
        corrected_words.extend(corrected_acronyms)
    return [
        {key: value for key, value in words.items() if key != "next_org_word_position"}
        for words in corrected_words
    ]
