
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

def train_bpe_tokenizer(texts, vocab_size=2000, save_path="model/bpe_tokenizer.json"):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"], vocab_size=vocab_size)
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(save_path)
    return tokenizer

def load_bpe_tokenizer(path="model/bpe_tokenizer.json"):
    return Tokenizer.from_file(path)

def tokenize_with_bpe(tokenizer, texts):
    return [" ".join(tokenizer.encode(t).tokens) for t in texts]
