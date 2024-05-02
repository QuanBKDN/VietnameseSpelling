VietAC
==============================

Spelling error correction is one of topics which have a long history in natural language processing. Although previous studies have achieved remarkable results, challenges still exist. In the Vietnamese language, a state-of-the-art method for the task infers a syllable’s context from its adjacent syllables. The method’s accuracy can be unsatisfactory, however, because the model may lose the context if two (or more) spelling mistakes stand near each other. In this project, we provide a method to correct Vietnamese spelling errors.

## Requirements

- Python 3.7+

## Installing


```shell
```
### Some install Issues
For some use-cases has installed old version [apex](https://pypi.org/project/apex/) will be raised following error:
```
RuntimeError: Failed to import transformers.trainer_seq2seq because of the following error (look up to see its traceback):
cannot import name 'UnencryptedCookieSessionFactoryConfig' from 'pyramid.session' (unknown location)
```

To solve this. You can use following command to uninstall apex:
```
pip uninstall apex
```

Or install the newer version of apex following [NVIDIA guides](https://github.com/NVIDIA/apex)

## Usage

### A. Inference

#### 1. Import

```python
from vietac import Corrector
```

#### 2. Init instance

To user our corrector, you must download our model and dictionary first.
To download our model for the first time to default folder **/temp/**:
```python
corrector = Corrector(download_dictionary=True, download_model=True)
```
or you can use
```python
corrector = Corrector(download_dictionary=True, model_path="./model",
            download_model=True, dictionary_path="./dictionaries",)
```
to download model into the specific folder.

If the model and dictionary were downloaded already, then you can point their address to the corrector.
```python
corrector = Corrector(model_path="./model", dictionary_path="./dictionaries",)
```
or you can use
```corrector = Corrector()```
if our model and dictionary are at default folder **/temp/**.
#### 3. Predict

**Predict on single text**

```python
text = corrector.infer("định nghĩa gia trị cực tiểu")
print(text)
```

**Predict on list of text**

```python
text = corrector.infer(["dinh nghia gia tri cuc tieu", "dhinh nghĩa gias trị cực dai"])
print(text)
```

### B. Training

#### 1. Prepare training data
We provide a create_training_data pipeline to create training data from raw text.
```python
from vietac import create_training_data

create_training_data(data_path="path to data file [txt, csv]",
                     save_path="path to save training data [csv]",
                     column_name="column name is needed for csv file")
```

#### 2. Train model
You can change the training configuration by replace the value in the configuration file. 
```python
from vietac import Trainer

trainer = Trainer(
    train_df=train_df,
    valid_df=valid_df,
    config_path="Path to config file"
)

trainer.train()
```


