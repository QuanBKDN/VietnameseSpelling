ORIGIN_ERROR_TEXT = {
    "long_text": "Sinh học phân tử là một phân ngành sinh học nhằm <error> nhiên </error> cứu <error> cơở </error> "
    "phân tử của các hoạt động sinh học bên trong và giữ các tế bào, bao gồm các quá trình "
    "<error> tong </error> hợp, <error> biêns </error> đổi, cơ <error> chees </error>,"
    " và các tương tác <error> phan </error> tử",
    "short_text": "<error> Co </error> quan cafe nào ngon <error> boo </error> <error> re </error> <error> ko </error>",
}

CORRECTED_ERROR_TEXT = {
    "long_text": "Sinh học phân tử là một phân ngành sinh học nhằm nghiên cứu cơ cấu phân tử của các hoạt động"
    " sinh học bên trong và giữ các tế bào, bao gồm các quá trình tong hợp, biến đổi, cơ ches,"
    " và các tương tác phân tử",
    "short_text": "Cơ quan cafe nào ngon lành cả",
}

ARRANGED_TEXT = {
    "long_text": [
        {"origin": "nhiên", "corrected": "nghiên", "start": 49, "end": 54},
        {"origin": "cơở", "corrected": "cơ cấu", "start": 59, "end": 62},
        {"origin": "tong", "corrected": "tong", "start": 149, "end": 153},
        {"origin": "biêns", "corrected": "biến", "start": 159, "end": 164},
    ],
    "short_text": [
        {"origin": "Co", "corrected": "Cơ", "start": 0, "end": 2},
        {"origin": "boo re ko", "corrected": "lành cả", "start": 22, "end": 31},
    ],
}

TEST_CASE_IS_THERE_CONTINUE_ERROR = {
    "Not have continue <error> error </error>": False,
    "Have <error> continue </error> <error> error </error>": True,
}

TEST_CASE_IS_START_WITH_ERROR = {
    "<error> Is </error> start with error": True,
    "Not start with <error> error </error>": False,
}

TEST_CASE_IS_END_WITH_ERROR = {
    "<error> Not </error> end with error": False,
    "Not <error> end </error> with error": False,
    "End with <error> error </error>": True,
}

TEST_CASE_IS_THERE_ERROR = {"<error> Is </error> there error": True, "Not error": False}

TEST_CASE_REMOVE_CONTINUE_ERROR_TAG = {
    "Have <error> continue </error> <error> error </error>": "Have <error> continue error </error>"
}

TEST_CASE_EXTRACT_CLEAN_STARTING_TEXT = {
    "<error> Is </error> start with error": [],
    "Not start with <error> error </error>": ["Not", "start", "with"],
}

TEST_CASE_EXTRACT_CLEAN_ENDING_TEXT = {
    "<error> Is </error> end with error": ["end", "with", "error"],
}

TEST_CASE_EXTRACT_ERROR_WORDS = {
    "Not have continue <error> error </error>": ["error"],
    "Have <error> continue </error> <error> error </error>": ["continue", "error"],
}

TEST_CASE_EXTRACT_NON_ERROR_WORDS = {
    "Error <error> in </error> remain <error> text </error>": ["remain"]
}

TEST_CASE_EXTRACT_REMAIN_TEXT_TO_ARRANGE = {
    (ORIGIN_ERROR_TEXT["long_text"], CORRECTED_ERROR_TEXT["long_text"]): [
        "nghiên",
        "cứu",
        "cơ",
        "cấu",
        "phân",
        "tử",
        "của",
        "các",
        "hoạt",
        "động",
        "sinh",
        "học",
        "bên",
        "trong",
        "và",
        "giữ",
        "các",
        "tế",
        "bào,",
        "bao",
        "gồm",
        "các",
        "quá",
        "trình",
        "tong",
        "hợp,",
        "biến",
        "đổi,",
        "cơ",
        "ches,",
        "và",
        "các",
        "tương",
        "tác",
        "phân",
    ],
    (ORIGIN_ERROR_TEXT["short_text"], CORRECTED_ERROR_TEXT["short_text"]): [
        "Cơ",
        "quan",
        "cafe",
        "nào",
        "ngon",
        "lành",
        "cả",
    ],
}

TEST_CASE_CHECK_AND_EXTRACT_CLEAN_STARTING_TEXT = {
    "Have <error> continue </error> <error> error </error>": ["Have"],
    "Not start with <error> error </error>": ["Not", "start", "with"],
}

TEST_CASE_CHECK_AND_EXTRACT_CLEAN_ENDING_TEXT = {"Clean <error> ending </error> text": ["text"]}

TEST_CASE_ADD_PREDICTED_AND_ORIGIN_TO_DICT = {
    ("origin word", "corrected word"): {"corrected": ["corrected word"], "origin": "origin word"}
}

TEST_CASE_GET_ERROR_INDEX = {
    "Hôm qua em đi học": [],
    "Hôm <error> qua </error> em đi <error> học </error>": [
        {"start": 4, "end": 7},
        {"start": 14, "end": 17},
    ],
    "<error> Hôm </error> qua em <error> đi </error> học <error> mẫu </error> giáo": [
        {"start": 0, "end": 3},
        {"start": 11, "end": 13},
        {"start": 18, "end": 21},
    ],
}
