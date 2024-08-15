import os
import torch
from torch.utils.data import random_split
from torch.distributed import init_process_group, destroy_process_group
from BERT import BERT, BERTLM
from trainer import BERTTrainer
from BERT_dataset import BERTDataset
import time
import pickle
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
DATA_LOADER_WORKERS = 4
BATCH_SIZE = 64
MAX_LEN =64
EPOCHS = 5

with open('./data/pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)

tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main():
    ddp_setup()
    dataset = BERTDataset(pairs, seq_len=MAX_LEN, tokenizer=tokenizer)

    bert_model = BERT(
        vocab_size=len(tokenizer.vocab),
        d_model=768,
        n_layers=2,
        heads=12,
        dropout=0.1
    )
    bert_lm = BERTLM(bert_model, len(tokenizer.vocab))
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
        sampler=DistributedSampler(dataset)
    )
    bert_trainer = BERTTrainer(bert_lm, train_loader)
    epochs = EPOCHS
    times = {}
    for epoch in range(epochs):
        start_time = time.time()
        bert_trainer.train(epoch)
        epoch_time = start_time - time.time()
        key_name = 'epoch_{}'.format(epoch)
        times[key_name] = epoch_time
    print(times)
    print("total runtime: ", sum(times.values()))
    destroy_process_group()


if __name__ == '__main__':
    main()
