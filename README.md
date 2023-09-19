# SpanMarker
BERT, RoBERTa 및 ELECTRA와 같은 인코더를 사용하여 NER 모델을 훈련하기 위한 프레임워크

PL-Marker 논문을 기반으로 하는 SpanMarker는 접근성과 사용 편의성을 통해 기존의 틀을 깨고 있음

SpanMarker는 bert-base-cased 및 roberta-large 와 같은 많은 일반 인코더와 함께 작동하며 IOB, IOB2, BIOES, BILOU 또는 라벨 주석 체계를 사용하지 않는 데이터 세트와도 자동으로 작동함

### 인코더 옵션

입력 인수로 position_ids를 허용해야 하므로 DistilBERT, T5, DistilRoBERTa, ALBERT 및 BART는 사용할 수 없음

또한 대문자로 명명된 엔티티를 찾는 데 사용되기 때문에 대소문자가 구분되지 않은 모델을 사용하는 것은 일반적으로 권장되지 않음

- BERT-BASE-CASE
- BERT-LARGE-CASE
- ROBERTA-BASE
- ROBERTA-LARGE

## Initializing a SpanMarkerModel

SpanMarker 모델은 SpanMarkerModel.from_pretrained를 통해 초기화

### **인코더 옵션**

- **[prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)**
- **[prajjwal1/bert-mini](https://huggingface.co/prajjwal1/bert-mini)**
- **[prajjwal1/bert-small](https://huggingface.co/prajjwal1/bert-small)**
- **[prajjwal1/bert-medium](https://huggingface.co/prajjwal1/bert-medium)**
- **[bert-base-cased](https://huggingface.co/bert-base-cased)**
- **[bert-large-cased](https://huggingface.co/bert-large-cased)**
- **[roberta-base](https://huggingface.co/roberta-base)**
- **[roberta-large](https://huggingface.co/roberta-large)**

```jsx
from span_marker import SpanMarkerModel

model_name = "roberta-base"
model = SpanMarkerModel.from_pretrained(
    model_name,
    labels=labels,
    model_max_length=256,
    entity_max_length=6,
)
```

```jsx
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaModel: ['lm_head.dense.bias', 'lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.decoder.weight', 'lm_head.dense.weight', 'lm_head.layer_norm.bias']
- This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
```

새로운 작업을 위해 BertModel을 초기화 할 때 발생하는 경고

## Training

Transformers와 SpanMarker Trainer에서 TrainingArguments를 직접 가져올 수 있음

Trainer는 일부 작업을 단순화하는 🤗 Transformers Trainer의 하위 클래스이지만 그 외에는 일반 Trainer와 동일하게 작동

아래 스니펫은 기본값

```jsx
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="models/span-marker-roberta-base-conll03",
    learning_rate=1e-5,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    push_to_hub=False,
    logging_steps=50,
    fp16=True,
    warmup_ratio=0.1,
)
```

이제 🤗 Transformers Trainer를 초기화하는 것과 동일한 방식으로 SpanMarker Trainer 생성 가능

span_marker의 `trainer` 는 사용자가 설치한 로깅 도구를 사용하여 자동으로 로그 생성

```jsx
from span_marker import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"].select(range(2000)),
)
trainer.train()
```

```jsx
This SpanMarker model will ignore 0.097877% of all annotated entities in the train dataset. This is caused by the SpanMarkerModel maximum entity length of 6 words.
These are the frequencies of the missed entities due to maximum entity length out of 23499 total entities:
- 18 missed entities with 7 words (0.076599%)
- 2 missed entities with 8 words (0.008511%)
- 3 missed entities with 10 words (0.012767%)
```

```jsx
{'loss': 0.9477, 'learning_rate': 2.6519337016574586e-06, 'epoch': 0.03}
{'loss': 0.2025, 'learning_rate': 5.414364640883978e-06, 'epoch': 0.06}
{'loss': 0.1407, 'learning_rate': 8.176795580110498e-06, 'epoch': 0.08}
{'loss': 0.1291, 'learning_rate': 9.895126465144972e-06, 'epoch': 0.11}
{'loss': 0.0973, 'learning_rate': 9.58667489204195e-06, 'epoch': 0.14}
{'loss': 0.0737, 'learning_rate': 9.278223318938926e-06, 'epoch': 0.17}
{'loss': 0.0639, 'learning_rate': 8.969771745835904e-06, 'epoch': 0.19}
{'loss': 0.0539, 'learning_rate': 8.661320172732882e-06, 'epoch': 0.22}
{'loss': 0.0481, 'learning_rate': 8.352868599629858e-06, 'epoch': 0.25}
{'loss': 0.0489, 'learning_rate': 8.044417026526836e-06, 'epoch': 0.28}
```

```jsx
This SpanMarker model won't be able to predict 0.172563% of all annotated entities in the evaluation dataset. This is caused by the SpanMarkerModel maximum entity length of 6 words.
These are the frequencies of the missed entities due to maximum entity length out of 3477 total entities:
- 5 missed entities with 7 words (0.143802%)
- 1 missed entities with 10 words (0.028760%)
```

```jsx
{'eval_loss': 0.03809420391917229, 'eval_overall_precision': 0.8559068219633943, 'eval_overall_recall': 0.7527070529704419, 'eval_overall_f1': 0.8009965742759265, 'eval_overall_accuracy': 0.9548683524504692, 'eval_runtime': 13.4517, 'eval_samples_per_second': 153.661, 'eval_steps_per_second': 38.434, 'epoch': 0.28}
{'loss': 0.0379, 'learning_rate': 7.735965453423812e-06, 'epoch': 0.31}
{'loss': 0.039, 'learning_rate': 7.42751388032079e-06, 'epoch': 0.33}
{'loss': 0.0373, 'learning_rate': 7.119062307217768e-06, 'epoch': 0.36}
{'loss': 0.0362, 'learning_rate': 6.810610734114744e-06, 'epoch': 0.39}
{'loss': 0.0287, 'learning_rate': 6.502159161011722e-06, 'epoch': 0.42}
{'loss': 0.0283, 'learning_rate': 6.193707587908698e-06, 'epoch': 0.44}
{'loss': 0.0308, 'learning_rate': 5.885256014805676e-06, 'epoch': 0.47}
{'loss': 0.0266, 'learning_rate': 5.576804441702654e-06, 'epoch': 0.5}
{'loss': 0.0193, 'learning_rate': 5.26835286859963e-06, 'epoch': 0.53}
{'loss': 0.0163, 'learning_rate': 4.959901295496608e-06, 'epoch': 0.55}
{'eval_loss': 0.018327122554183006, 'eval_overall_precision': 0.9140995260663507, 'eval_overall_recall': 0.9031314018144572, 'eval_overall_f1': 0.9085823641984395, 'eval_overall_accuracy': 0.9804157977059437, 'eval_runtime': 13.537, 'eval_samples_per_second': 152.693, 'eval_steps_per_second': 38.192, 'epoch': 0.55}
{'loss': 0.0249, 'learning_rate': 4.651449722393585e-06, 'epoch': 0.58}
{'loss': 0.0225, 'learning_rate': 4.342998149290562e-06, 'epoch': 0.61}
{'loss': 0.0215, 'learning_rate': 4.0345465761875395e-06, 'epoch': 0.64}
{'loss': 0.0251, 'learning_rate': 3.726095003084516e-06, 'epoch': 0.67}
{'loss': 0.0186, 'learning_rate': 3.417643429981493e-06, 'epoch': 0.69}
{'loss': 0.0212, 'learning_rate': 3.10919185687847e-06, 'epoch': 0.72}
{'loss': 0.0166, 'learning_rate': 2.800740283775448e-06, 'epoch': 0.75}
{'loss': 0.0226, 'learning_rate': 2.492288710672425e-06, 'epoch': 0.78}
{'loss': 0.0162, 'learning_rate': 2.183837137569402e-06, 'epoch': 0.8}
```

```jsx
Loading cached processed dataset at ...
Loading cached processed dataset at ...
```

```jsx
{'loss': 0.0178, 'learning_rate': 1.8753855644663791e-06, 'epoch': 0.83}
{'eval_loss': 0.013812492601573467, 'eval_overall_precision': 0.9375370041444642, 'eval_overall_recall': 0.9268364062042728, 'eval_overall_f1': 0.9321559970566594, 'eval_overall_accuracy': 0.9858902502606882, 'eval_runtime': 13.6173, 'eval_samples_per_second': 151.793, 'eval_steps_per_second': 37.967, 'epoch': 0.83}
{'loss': 0.017, 'learning_rate': 1.566933991363356e-06, 'epoch': 0.86}
{'loss': 0.0164, 'learning_rate': 1.2584824182603333e-06, 'epoch': 0.89}
{'loss': 0.0202, 'learning_rate': 9.500308451573104e-07, 'epoch': 0.92}
{'loss': 0.0203, 'learning_rate': 6.415792720542875e-07, 'epoch': 0.94}
{'loss': 0.0188, 'learning_rate': 3.3312769895126467e-07, 'epoch': 0.97}
{'loss': 0.0175, 'learning_rate': 2.4676125848241828e-08, 'epoch': 1.0}
{'train_runtime': 326.1576, 'train_samples_per_second': 44.193, 'train_steps_per_second': 5.525, 'train_loss': 0.06725485075976323, 'epoch': 1.0}
```

## compute the model’s performance

```jsx
metrics = trainer.evaluate()
metrics
```

```jsx
Loading cached processed dataset at ...
Loading cached processed dataset at ...
```

```jsx
{'eval_loss': 0.01344103179872036,
 'eval_overall_precision': 0.9364892678623934,
 'eval_overall_recall': 0.9321041849575651,
 'eval_overall_f1': 0.9342915811088296,
 'eval_overall_accuracy': 0.9861183524504692,
 'eval_runtime': 13.1532,
 'eval_samples_per_second': 157.148,
 'eval_steps_per_second': 39.306,
 'epoch': 1.0}
```

## evaluate using the test set

```jsx
trainer.evaluate(dataset["test"], metric_key_prefix="test")
```

```jsx
{'test_loss': 0.027700792998075485,
 'test_overall_precision': 0.9039692701664532,
 'test_overall_recall': 0.9067889908256881,
 'test_overall_f1': 0.9053769350554182,
 'test_overall_accuracy': 0.9796867082332724,
 'test_runtime': 22.7367,
 'test_samples_per_second': 155.915,
 'test_steps_per_second': 39.012,
 'epoch': 1.0}
```

## save the model

```jsx
trainer.save_model("models/span-marker-roberta-base-conll03/checkpoint-final")
```

##
**If we put it all together into a single script, it looks something like this:**

```jsx
from datasets import load_dataset
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments

dataset = load_dataset("conll2003")
labels = dataset["train"].features["ner_tags"].feature.names

model_name = "roberta-base"
model = SpanMarkerModel.from_pretrained(model_name, labels=labels, model_max_length=256)

args = TrainingArguments(
    output_dir="models/span-marker-roberta-base-conll03",
    learning_rate=1e-5,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    push_to_hub=False,
    logging_steps=50,
    warmup_ratio=0.1,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"].select(range(8000)),
    eval_dataset=dataset["validation"].select(range(2000)),
)

trainer.train()
trainer.save_model("models/span-marker-roberta-base-conll03/checkpoint-final")
trainer.push_to_hub()

metrics = trainer.evaluate()
print(metrics)
```
