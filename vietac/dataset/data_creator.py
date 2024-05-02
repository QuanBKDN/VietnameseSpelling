from random import choices, randint, uniform

import numpy as np
import pandas as pd
from tqdm import tqdm

from vietac.dataset.augment_error import (
    ErrorsWrapper,
    add_accents,
    add_error_tag,
    add_misspell_head_consonant,
    add_misspell_tail_consonant,
    add_telex_errors,
    create_error_with_control,
    delete_char,
    insert_char,
    remove_accents,
    remove_diacritics,
)
from vietac.dataset.clean import clean_input_text
from vietac.utils import logger


def create_training_data(data_path: str, save_path: str, column_name: str = "text"):
    """
    Create training data from raw text.
    Args:
        data_path: Path to data file.
        save_path: Path to save training data
        column_name: Column name if data file is a csv

    Returns: Save training data in save_path

    """
    if data_path[-3:] == "txt":
        with open(data_path, "r") as f:
            data = f.readlines()
        data = [clean_input_text(text) for text in data]
    elif data_path[-3:] == "csv":
        data = pd.read_csv(data_path)[column_name].to_list()
    else:
        raise Exception("Error: Data file must is text file or csv file!")

    main_error_list = [
        # remove special
        ErrorsWrapper([remove_diacritics, remove_accents]),
        # add telex
        ErrorsWrapper([add_telex_errors, delete_char]),
        ErrorsWrapper([add_telex_errors, insert_char]),
        # remove diacritic
        remove_diacritics,
        # remove accents
        remove_accents,
        # add misspell
        add_misspell_tail_consonant,
        add_misspell_head_consonant,
    ]

    secondary_error_list = [delete_char, insert_char, add_accents]
    error_distribution = np.random.normal(loc=2, size=len(data), scale=1)
    error_distribution = [r if r > 1 else uniform(1, 6) for r in error_distribution]

    origin_sentences = []
    error_sentences = []
    detect_targets = []
    correct_targets = []
    logger.info("Creating data ...")
    for i, sentence in enumerate(tqdm(data)):
        try:
            sentence = sentence.replace("ntn", "như thế nào")
            sentence = sentence.replace("đ/nghĩa", "định nghĩa")
            sentence = sentence.replace("'", "")
            sentence = sentence.replace("–", "-")
            sentence = sentence.replace(":", "")
            sentence = sentence.replace(";", "")

            while "  " in sentence:
                sentence = sentence.replace("  ", " ")
            sentence = sentence.strip()
            if sentence == "":
                continue
        except ValueError:
            continue
        error_ratio = error_distribution[i] / 10
        num_main_errors = randint(1, 3)
        num_secondary_errors = randint(1, 2)

        main_errors = choices(main_error_list, k=num_main_errors)
        secondary_errors = choices(secondary_error_list, k=num_secondary_errors)
        errors_list = main_errors + secondary_errors
        error_type_ratio = [
            ((1 - 0.2 * num_secondary_errors) / num_main_errors) for _ in range(num_main_errors)
        ] + [
            (0.2 * num_secondary_errors) / num_secondary_errors
            for _ in range(num_secondary_errors)
        ]

        error_sentence = create_error_with_control(
            sentence=sentence,
            error_ratio=error_ratio,
            functions_create_error=errors_list,
            ratios_error_type=error_type_ratio,
        )
        error_tagged, target = add_error_tag(sentence, error_sentence)

        error_sentences.append(error_sentence)
        detect_targets.append(error_tagged)
        correct_targets.append(target)
        origin_sentences.append(sentence)

    pd.DataFrame(
        {
            "corrected": origin_sentences,
            "input_text": error_sentences,
            "detected": detect_targets,
            "new_target": correct_targets,
        }
    ).to_csv(save_path, index=False)
    logger.info(f"Data saved at {save_path}")
