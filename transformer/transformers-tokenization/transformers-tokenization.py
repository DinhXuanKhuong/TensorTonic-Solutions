import numpy as np
from typing import List, Dict
import string

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        # pass
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token
        

        texts = [x.lower() for x in texts]

        vocab = set()

        for text in texts:
            vocab.update(set(text.split(" ")))
        
        vocab = sorted(list(vocab))
            
        for i, word in enumerate(vocab):
            self.word_to_id[word] = i + 4
            self.id_to_word[i + 4] = word
        
        self.vocab_size = len(self.word_to_id)
            
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        text = text.lower()
        word_list = text.split(" ")
        
        id_list = []
        for word in word_list:
            if word == "":
                # id_list.append(0)
                continue
            if word not in self.word_to_id:
                id_list.append(1)
            else:
                id_list.append(self.word_to_id[word])
        return id_list
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        word_list = []

        for id in ids:
            if id in self.id_to_word:
                word_list.append(self.id_to_word[id])
            else:
                word_list.append(self.id_to_word[1])
                
        
        text = " ".join(word_list)
        return text