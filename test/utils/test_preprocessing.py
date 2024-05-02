from test.fixtures.utils.preprocessing import (
    ARRANGED_TEXT,
    CORRECTED_ERROR_TEXT,
    ORIGIN_ERROR_TEXT,
    TEST_CASE_ADD_PREDICTED_AND_ORIGIN_TO_DICT,
    TEST_CASE_CHECK_AND_EXTRACT_CLEAN_ENDING_TEXT,
    TEST_CASE_CHECK_AND_EXTRACT_CLEAN_STARTING_TEXT,
    TEST_CASE_EXTRACT_CLEAN_ENDING_TEXT,
    TEST_CASE_EXTRACT_CLEAN_STARTING_TEXT,
    TEST_CASE_EXTRACT_ERROR_WORDS,
    TEST_CASE_EXTRACT_NON_ERROR_WORDS,
    TEST_CASE_EXTRACT_REMAIN_TEXT_TO_ARRANGE,
    TEST_CASE_GET_ERROR_INDEX,
    TEST_CASE_IS_END_WITH_ERROR,
    TEST_CASE_IS_START_WITH_ERROR,
    TEST_CASE_IS_THERE_CONTINUE_ERROR,
    TEST_CASE_IS_THERE_ERROR,
    TEST_CASE_REMOVE_CONTINUE_ERROR_TAG,
)

from vietac.utils.preprocessing import (
    _is_end_with_error,
    _is_start_with_error,
    _is_there_continue_error,
    _is_there_error,
    _remove_continue_error_tag,
    add_predicted_and_origin_to_dict,
    arrange_corrected_to_origin,
    check_and_extract_clean_ending_text,
    check_and_extract_clean_starting_text,
    extract_clean_ending_text,
    extract_clean_starting_text,
    extract_clean_text,
    extract_error_words,
    extract_non_error_words,
    extract_remain_text_to_arrange,
    get_list_of_error_index,
    replace_acronym,
)


single_acronym_test_cases = {
    "t là svtt của cty ftech": "t là sinh viên thực tập của công ty ftech",
    "nêu đ/nghĩa của ptrình": "nêu định nghĩa của phương trình",
    "hỉu chết liền": "hiểu chết liền",
}

multiple_acronym_test_cases = {"cái j z má": "cái gì vậy má", "đang ở mô đó": "đang ở đâu đó"}

mix_acronym_test_cases = {
    "m đang làm svtt ở mô đó?": "m đang làm sinh viên thực tập ở đâu đó?",
    "xg sống t khum ổn m ơi": "xương sống t không ổn m ơi",
    "bài nớ tìm bcnn dc k?": "bài đó tìm bội chung nhỏ nhất được không?",
}

arrange_output_test_cases = {
    (ORIGIN_ERROR_TEXT["long_text"], CORRECTED_ERROR_TEXT["long_text"]): ARRANGED_TEXT[
        "long_text"
    ],
    (ORIGIN_ERROR_TEXT["short_text"], CORRECTED_ERROR_TEXT["short_text"]): ARRANGED_TEXT[
        "short_text"
    ],
}


def test_single_acronym():
    for raw_input, preprocessed_output in single_acronym_test_cases.items():
        assert replace_acronym(raw_input) == preprocessed_output


def test_multiple_acronym():
    for raw_input, preprocessed_output in multiple_acronym_test_cases.items():
        assert replace_acronym(raw_input, non_single_acronym=True) == preprocessed_output


def test_mix_acronym():
    for raw_input, preprocessed_output in mix_acronym_test_cases.items():
        explication = replace_acronym(raw_input, non_single_acronym=True)
        explication = replace_acronym(explication, non_single_acronym=False)
        assert explication == preprocessed_output


def test_arrange_corrected_to_origin():
    for texts_input, expected_arranged_output in arrange_output_test_cases.items():
        origin_error_text, corrected_error_text = texts_input

        arranged_text = arrange_corrected_to_origin(
            predicted_error_text=origin_error_text, corrected_text=corrected_error_text
        )
        differences = [
            difference_item
            for difference_item in arranged_text + expected_arranged_output
            if difference_item not in arranged_text
            or difference_item not in expected_arranged_output
        ]
        assert not differences


def test_is_there_error():
    for texts_input, expected_output in TEST_CASE_IS_THERE_ERROR.items():
        output = _is_there_error(texts_input)
        assert output == expected_output


def test_is_start_with_error():
    for texts_input, expected_output in TEST_CASE_IS_START_WITH_ERROR.items():
        output = _is_start_with_error(texts_input)
        assert output == expected_output


def test_is_end_with_error():
    for texts_input, expected_output in TEST_CASE_IS_END_WITH_ERROR.items():
        output = _is_end_with_error(texts_input)
        assert output == expected_output


def test_remove_continue_error_tag():
    for texts_input, expected_output in TEST_CASE_REMOVE_CONTINUE_ERROR_TAG.items():
        output = _remove_continue_error_tag(texts_input)
        assert output == expected_output


def test_extract_clean_starting_text():
    for texts_input, expected_output in TEST_CASE_EXTRACT_CLEAN_STARTING_TEXT.items():
        output = extract_clean_starting_text(texts_input)
        assert output == expected_output


def test_extract_clean_ending_text():
    for texts_input, expected_output in TEST_CASE_EXTRACT_CLEAN_ENDING_TEXT.items():
        output = extract_clean_ending_text(texts_input)
        assert output == expected_output


def test_extract_error_words():
    for texts_input, expected_output in TEST_CASE_EXTRACT_ERROR_WORDS.items():
        output = extract_error_words(texts_input)
        assert output == expected_output


def test_extract_non_error_words():
    for texts_input, expected_output in TEST_CASE_EXTRACT_NON_ERROR_WORDS.items():
        output = extract_non_error_words(texts_input)
        assert output == expected_output


def test_extract_is_there_error():
    for texts_input, expected_output in TEST_CASE_IS_THERE_ERROR.items():
        output = _is_there_error(texts_input)
        assert output == expected_output


def test_check_and_extract_clean_ending_text():
    for texts_input, expected_output in TEST_CASE_CHECK_AND_EXTRACT_CLEAN_ENDING_TEXT.items():
        output = check_and_extract_clean_ending_text(texts_input)
        assert output == expected_output


def test_check_and_extract_clean_starting_text():
    for texts_input, expected_output in TEST_CASE_CHECK_AND_EXTRACT_CLEAN_STARTING_TEXT.items():
        output = check_and_extract_clean_starting_text(texts_input)
        assert output == expected_output


def test_is_there_continue_error():
    for texts_input, expected_output in TEST_CASE_IS_THERE_CONTINUE_ERROR.items():
        output = _is_there_continue_error(texts_input)
        assert output == expected_output


def test_add_predicted_and_origin_to_dict():
    for texts_input, expected_output in TEST_CASE_ADD_PREDICTED_AND_ORIGIN_TO_DICT.items():
        origin_word, corrected_words = texts_input
        output = add_predicted_and_origin_to_dict(
            origin_word=origin_word, corrected_words=[corrected_words]
        )
        assert output == expected_output


def test_extract_remain_text_to_arrange():
    for texts_input, expected_output in TEST_CASE_EXTRACT_REMAIN_TEXT_TO_ARRANGE.items():
        predicted_error_text, corrected_text = texts_input
        list_of_corrected_text = corrected_text.split(" ")
        clean_starting_text, clean_ending_text = extract_clean_text(predicted_error_text)
        output = extract_remain_text_to_arrange(
            corrected_text=list_of_corrected_text,
            clean_starting_text=clean_starting_text,
            clean_ending_text=clean_ending_text,
        )
        assert output == expected_output


def test_get_list_of_error_index():
    for texts_input, expected_output in TEST_CASE_GET_ERROR_INDEX.items():
        output = get_list_of_error_index(texts_input)
        differences = [
            difference_item
            for difference_item in output + expected_output
            if difference_item not in output or difference_item not in expected_output
        ]
        assert not differences
