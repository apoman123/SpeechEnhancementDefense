import argparse

from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from datasets import load_from_disk, Audio, concatenate_datasets, load_dataset, DatasetDict
import numpy as np

from RCNN import KWSModel

def zero_mean_normalization(data):
    mean = torch.mean(data, dim=-1, keepdim=True).expand(data.shape)
    std = torch.std(data, dim=-1, keepdim=True).expand(data.shape)
    data = (data - mean) / std
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--dataset_path", type=str)
    # training
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--resume_path", type=str)
    return parser.parse_args()
    


class Collator():
    def __init__(self):
        pass
        
    def __call__(self, batch):
        global transform
        input_data = [torch.from_numpy(data['audio']['array']).float() for data in batch]
        lens = [data.shape[-1] for data in input_data]
        max_len = max(lens)
        input_data = [
            torch.cat([
                data,
                torch.zeros(max_len - data.shape[-1])
            ]) for data in input_data
        ]
        input_data = torch.stack(input_data, dim=0)
        input_data = zero_mean_normalization(input_data)
        labels = torch.tensor([data['label'] for data in batch])
        return {"input_data": transform(input_data), "labels": labels}
    
class KWSModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        result = model(inputs['input_data'])
        loss = F.cross_entropy(result, inputs['labels'])
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        result = model(inputs['input_data'])
        loss = F.cross_entropy(result, inputs['labels'])
        return (loss.detach(), result.detach(), inputs['labels'].detach())

    def compute_metrics(eval_pred):
        raise
        logits, labels = eval_pred.prediction, eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(predictions, labels),
            "recall": recall_score(predictions, labels), 
            "precision": precision_score(predictions, labels),
            "f1": f1_score(predictions, labels)
        }
        
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(predictions, labels),
        # "recall": recall_score(predictions, labels), 
        # "precision": precision_score(predictions, labels),
        # "f1": f1_score(predictions, labels)
    }

if __name__ == "__main__":
    args = parse_args()
    transform = MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=100,
        n_mels=40
    )
    
    qkws = load_from_disk(args.dataset_path).cast_column("audio", Audio(sampling_rate=16000))
    augmented_training_set = load_from_disk(args.dataset_path).cast_column("audio", Audio(sampling_rate=16000))
    training_set = concatenate_datasets([qkws['train'], augmented_training_set['train']])
    valid_set = concatenate_datasets([qkws['valid'], augmented_training_set['valid']])
    # training_set = load_from_disk("/data/nas07/PersonalData/apoman123/vctk_mpsenet_augmented_training_set")
    # testing_set = load_from_disk("/data/nas07/PersonalData/apoman123/vctk_mpsenet_augmented_testing_set")
    collator = Collator()
    
    model = KWSModel()
    model_size = sum(t.numel() for t in model.parameters())
    print(model)
    print(f"KWSModel size: {model_size/1000**2:.1f}M parameters")
    
    training_args = TrainingArguments(
        output_dir=args.save_path,
        logging_dir=args.log_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        warmup_steps=1000,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=1,
        remove_unused_columns=False,
        do_eval=True,
        do_predict=True
    )
    
    trainer = KWSModelTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=training_set,
        eval_dataset=valid_set,
        compute_metrics=compute_metrics
    )

    trainer.train(resume_from_checkpoint=args.resume_path)
    





    


