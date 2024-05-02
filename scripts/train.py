import pandas as pd
from sklearn.model_selection import train_test_split
from vietac.trainer import Trainer


if __name__ == "__main__":
    df = pd.read_csv("data/data_v3/va_train_97k.csv")
    train_df, valid_df = train_test_split(df, train_size=0.9, shuffle=True, random_state=19)

    trainer = Trainer(
        train_df=train_df, valid_df=valid_df, config_path="vietac/configs/configs.yaml"
    )

    trainer.train()
