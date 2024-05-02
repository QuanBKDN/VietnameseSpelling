import pandas as pd
from vietac.dataset import create_training_data


if __name__ == "__main__":
    news_text_list = pd.read_csv("data/data_v3/news_clean_text_500k.csv")["text"].to_list()
    create_training_data(
        data_path="data/data_v3/news_clean_text_500k.csv",
        save_path="data/data_v3/va_train_97k.csv",
    )
