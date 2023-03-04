import os
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


# Inspired by https://pytorch.org/tutorials/beginner/translation_transformer.html

def dataset_path_from_config(config, dataset_type, language):
    """Gets path to dataset file for dataset type and language

    Args:
        config (dict): configuration for creating a dataset, see config.yaml
        dataset_type (str): train, test or val
        language (str): src or dst

    Returns:
        str: path to dataset file
    """
    return os.path.join(config["datadir"], config["dataset"][dataset_type][language])

def vocab_path_from_config(config):
    """Gets path to vocab file for dataset type

    Args:
        config (dict): configuration for creating a dataset, see config.yaml

    Returns:
        _type_: path to vocab file
    """
    return os.path.join(config["vocabdir"], "vocab.pth")

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


class TextDataset(Dataset):
    UNK_IDX = 0
    PAD_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    def __init__(self, config, dataset_type, rebuild=False):
        """Creates a dataset for seq2seq task

        Args:
            config (dict): configuration for creating a dataset, see config.yaml
            dataset_type (str): train, val or test
            rebuild (bool, optional): Whether to rebuild vocab or use version from disk. Used only for train dataset. Defaults to False.
        """
        super().__init__()
        
        assert not (rebuild and dataset_type != "train"), "You can rebuild vocab only within train dataset"

        self.dataset_type = dataset_type

        paths = []
        src_path = dataset_path_from_config(config, dataset_type, "src")
        paths.append(src_path)

        if not self.is_test:
            dst_path = dataset_path_from_config(config, dataset_type, "dst")
            paths.append(dst_path)

        self.src_lang = config["language"]["src"]
        self.dst_lang = None

        if not self.is_test:
            self.dst_lang = config["language"]["dst"]

        self.texts = {}

        # Load texts from dataset
        for ln, path in zip(self.get_languages(), paths):
            self.texts[ln] = TextDataset.texts_from_dataset(path)

        if not self.is_test:
            assert len(self.texts[self.src_lang]) == len(self.texts[self.dst_lang]), "Size of src and dst datasets must match"

        self.tokenizers = {}

        # Create tokenizers
        for ln in self.get_languages():
            self.tokenizers[ln] = TextDataset.create_tokenizer(ln)

        vocab_path = vocab_path_from_config(config)

        self.vocabs = {}

        # Get vocab
        if os.path.exists(vocab_path) and not rebuild:
            logging.info(f"Loaded vocab {dataset_type} {self.src_lang}/{self.dst_lang if not self.is_test else '*'} from {vocab_path}")
            self.vocabs = torch.load(vocab_path)
        else:
            for ln in self.get_languages():
                self.vocabs[ln] = TextDataset.create_vocab(self.tokenizers[ln], self.texts[ln])
            
            if not os.path.exists(os.path.dirname(vocab_path)):
                os.mkdir(os.path.dirname(vocab_path))
            torch.save(self.vocabs, vocab_path)

        self.transforms = {}
        for ln in self.get_languages():
            self.transforms[ln] = sequential_transforms(
                self.tokenizers[ln], self.vocabs[ln], TextDataset.tensor_transform
            )
    
    @property
    def is_test(self):
        return self.dataset_type == "test"

    @staticmethod
    def texts_from_dataset(dataset_path):
        """Returns list of sentences from dataset

        Args:
            dataset_path (str): path to dataset
        
        Returns:
            list: list of sentences
        """
        with open(dataset_path, encoding="utf-8") as f:
            return [line.rstrip() for line in f.readlines()]

    @staticmethod
    def create_tokenizer(language):
        """Creates tokenizer for language

        Args:
            language (str): e.g en, ge

        Returns:
            Any: tokenizer for language
        """
        return get_tokenizer(None, language=language)

    @staticmethod
    def yield_token(tokenizer, texts):
        """Generator for yielding tokens

        Args:
            tokenizer (Any): torch tokenizer
            texts (list): list of sentences

        Yields:
            list: sentence splitted by space
        """
        for line in tqdm(texts, desc="Building vocab"):
            sentence = tokenizer(line)
            for token in sentence:
                yield token

    @staticmethod
    def create_vocab(tokenizer, texts):
        """Creates vocab from dataset for specific language

        Args:
            tokenizer (Any): transforms given text, e.g splits by space
            texts (list): list of sentences
        Returns:
            torchtext.Vocab: vocabulary built from dataset
        """
        special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
        
        vocab = build_vocab_from_iterator([TextDataset.yield_token(tokenizer, texts)],
                                          min_freq=1, special_first=True, specials=special_symbols)
        vocab.set_default_index(TextDataset.UNK_IDX)

        return vocab
    
    @property
    def src_vocab(self):
        return self.vocabs[self.src_lang]

    @property
    def dst_vocab(self):
        assert not self.is_test, "There is no dst vocab for test dataset"
        return self.vocabs[self.dst_lang]

    def get_languages(self):
        """Returns a tuple of languages in translation task

        Returns:
            tuple: src and dst languages
        """
        if not self.is_test:
            return self.src_lang, self.dst_lang
        return (self.src_lang, )

    def __len__(self):
        """Returns length of a dataset
        """
        return len(self.texts[self.src_lang])

    @staticmethod
    def tensor_transform(indices):
        """Adds bos and eos indices at the beginning and at end of a sentence

        Args:
            indices (list): list of indices after vocab transform

        Returns:
            torch.Tensor: tensor of indices
        """
        return torch.cat((torch.tensor([TextDataset.BOS_IDX]),
                          torch.tensor(indices),
                          torch.tensor([TextDataset.EOS_IDX])))

    def getitem(self, index, language):
        """Returns sentence after transformations in a dataset of specific language

        Args:
            index (int): index of sentence
            language (str): language of sentence

        Returns:
            torch.tensor: sentence after all transformations
        """
        return self.transforms[language](self.texts[language][index])

    def __getitem__(self, index):
        """Returns sentence after transformations in a dataset of specific language

        Args:
            index (int): index of sentence

        Returns:
            torch.tensor: sentence after all transformations
        """
        return tuple(self.getitem(index, ln) for ln in self.get_languages())

    @staticmethod
    def get_collate_fn(is_test=False):
        def collate_fn(batch):
            src_batch = pad_sequence([b[0] for b in batch], padding_value=TextDataset.PAD_IDX, batch_first=True)

            if is_test:
                return (src_batch, None)

            dst_batch = pad_sequence([b[1] for b in batch], padding_value=TextDataset.PAD_IDX, batch_first=True)
            
            return src_batch, dst_batch
        
        return collate_fn