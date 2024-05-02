import math

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from vietac.dataset.preprocess import create_training_data
from vietac.utils import logger
from vietac.utils.configs import read_config


class Trainer(Seq2SeqTrainer):
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        config_path: str = None,
        pretrain_model: str = None,
        training_args: Seq2SeqTrainingArguments = None,
    ):
        """
        Create a trainer based on transformer Seq2SeqTrainer
        Args:
            train_df: training dataframe
            valid_df: validation dataframe
            config_path: path to config file
            pretrain_model: pretrained_model_name_or_path (`required if config_path was not set`)
            training_args: Seq2SeqTrainingArguments (`required if config_path was not set`)
        """

        if training_args and pretrain_model:
            args = training_args
            self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_model)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
        elif config_path:
            config = read_config(config_path)
            log_step = math.ceil(math.ceil(len(train_df) / config.train.batchsize) * 0.01)
            save_step = math.ceil(
                math.ceil(len(train_df) / config.train.batchsize) * config.train.num_epochs * 0.1
            )
            args = Seq2SeqTrainingArguments(
                output_dir=config.train.save_path,
                evaluation_strategy="steps",
                eval_steps=save_step,
                logging_strategy="steps",
                logging_steps=log_step,
                save_strategy="steps",
                save_steps=save_step,
                learning_rate=config.train.learning_rate,
                per_device_train_batch_size=config.train.batchsize,
                per_device_eval_batch_size=config.train.batchsize,
                weight_decay=config.train.weight_decay,
                save_total_limit=config.train.limit_save,
                num_train_epochs=config.train.num_epochs,
                predict_with_generate=True,
                report_to=["wandb"],
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.train.pretrained_model)
            self.tokenizer = AutoTokenizer.from_pretrained(config.train.pretrained_model)
        else:
            raise Exception("config_path or (pretrain_model, pretrain_model) is required")

        self._add_special_token()

        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        logger.info("Prepare training data")
        train_inputs = create_training_data(train_df, self.tokenizer)
        logger.info("Prepare validation data")
        valid_inputs = create_training_data(valid_df, self.tokenizer)
        train_dataset = Dataset.from_dict(train_inputs)
        valid_dataset = Dataset.from_dict(valid_inputs)

        super().__init__(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

    def _add_special_token(self):
        new_tokens = ["<error>", "</error>", "</correct>", "<corrected>", "<non_error>"]
        new_tokens = set(new_tokens) - set(self.tokenizer.vocab.keys())
        self.tokenizer.add_tokens(list(new_tokens), special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
