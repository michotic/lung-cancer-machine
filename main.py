from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from torchvision import transforms
from datasets import load_dataset
import numpy as np
import evaluate
import torch

PATH_TO_DATASET = "data"
MODEL_PATH = "checkpoint-76"
# MODEL_PATH = "google/vit-base-patch16-224"
# "google/vit-base-patch16-224"
# "cancer_model"


def get_dataset(folder_path):
    """
    Loads dataset of CT scan images

    LABELS:
    0-> "adeno carcinoma" (cancer type A)
    1-> "large cell carcinoma" (cancer type B)
    2-> "normal" (no cancer)
    3-> "squamous cell carcinoma" (cancer type C)
    """
    _label2id = {
        "adeno carcinoma": 0,
        "large cell carcinoma": 1,
        "normal": 2,
        "squamous cell carcinoma": 3,
    }
    _id2label = {
        0: "adeno carcinoma",
        1: "large cell carcinoma",
        2: "normal",
        3: "squamous cell carcinoma",
    }
    dataset = load_dataset("imagefolder", data_dir=folder_path)

    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def preprocess(samples):
        try:
            samples["pixel_values"] = image_processor.preprocess(
                images=samples["image"], return_tensors="pt"
            ).pixel_values
        except:
            samples["pixel_values"] = image_processor.preprocess(
                images=[img.convert("RGB") for img in samples["image"]],
                return_tensors="pt",
            ).pixel_values
        del samples["image"]
        return samples

    dataset = dataset.map(preprocess, batched=True, batch_size=3)
    print("Dataset loaded successfully.")
    return dataset, _id2label, _label2id, image_processor


def setup_model(base_model_path, _id2label, _label2id, _labels):
    model = ViTForImageClassification.from_pretrained(
        base_model_path,
        num_labels=len(_labels),
        id2label=_id2label,
        label2id=_label2id,
        ignore_mismatched_sizes=True,
    )

    print("Model loaded successfully.")
    return model


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def finetune_model(model, dataset, image_processor):
    data_collator = DefaultDataCollator()

    for param in model.base_model.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        use_mps_device=True,
        output_dir="cancer_trainer",
        remove_unused_columns=False,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=1e-3,
        optim="adamw_torch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        warmup_ratio=0.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.00,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained("cancer_model")


def validate_model(model, test_dataset):
    model.to("cpu")
    model.eval()
    correct = [0, 0, 0, 0]
    incorrect = [0, 0, 0, 0]
    for sample in test_dataset:
        inputs = torch.as_tensor(sample["pixel_values"], dtype=torch.float32)
        inputs = inputs[None, :, :, :]
        with torch.no_grad():
            logits = model(inputs).logits
        predicted_label = logits.argmax(-1).item()
        if predicted_label == sample["label"]:
            correct[sample["label"]] += 1
        else:
            incorrect[sample["label"]] += 1
            print(
                "incorrect prediction... predicted:",
                model.config.id2label[predicted_label],
                "expected:",
                model.config.id2label[sample["label"]],
            )

    for i in range(len(correct)):
        print("\n", model.config.id2label[i])
        print("right:", correct[i], " wrong:", incorrect[i])
        print("accuracy:", correct[i] / (correct[i] + incorrect[i]))
    print("\noverall")
    print("right:", sum(correct), "wrong:", sum(incorrect))
    print("accuracy:", sum(correct) / (sum(correct) + sum(incorrect)))


if __name__ == "__main__":
    dataset, id2label, label2id, image_processor = get_dataset(PATH_TO_DATASET)
    labels = label2id.keys()
    model = setup_model(MODEL_PATH, id2label, label2id, labels)
    finetune_model(model, dataset, image_processor)
    validate_model(model, dataset["validation"])
