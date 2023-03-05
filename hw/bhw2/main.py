import os
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import read_config, config_str
from data import TextDataset
from model import translate
from trainer import Trainer

import gc

logging.basicConfig(
    handlers=[logging.FileHandler("debug.log", mode='w'), logging.StreamHandler()],
    level=logging.INFO, 
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
 )

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def main():
    torch.cuda.empty_cache()
    gc.collect()

    config = read_config()
    
    logging.info(config_str(config))

    batch_size = config["batch_size"]
    
    train_dataset = TextDataset(config, "train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TextDataset.get_collate_fn(), num_workers=config["num_workers"], pin_memory=True)

    val_dataset = TextDataset(config, "val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=TextDataset.get_collate_fn(), num_workers=config["num_workers"], pin_memory=True)

    max_length = 128
    vocab_sizes = len(train_dataset.src_vocab), len(train_dataset.dst_vocab)     

    logging.info("Vocab sizes {}".format(vocab_sizes))

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Device {}".format(device))

    trainer = Trainer(config, vocab_sizes, max_length, device, run_name="test")
    trainer.train(config, train_loader, val_loader)

    trainer.load_from_checkpoint(epoch=10)

    test_dataset = TextDataset(config, "test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=TextDataset.get_collate_fn(is_test=True))
    dst_vocab = train_dataset.vocabs["en"]

    translated = []

    for src, _ in tqdm(test_loader):
        dst_tokens = list(translate(trainer.model, src[0], device).cpu().numpy())
        sentence = " ".join(dst_vocab.lookup_tokens(dst_tokens)).replace("<bos>", "").replace("<eos>", "").strip()
        translated.append(sentence)

    with open(config["outputfile"], "w") as f:
        for sentence in translated:
            f.write(sentence + "\n")

if __name__ == "__main__":
    main()