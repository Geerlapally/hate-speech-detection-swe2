
import random

char_map = {
    'a': ['@', '4'],
    'e': ['3'],
    'i': ['1', '!'],
    'o': ['0'],
    's': ['$', '5'],
    't': ['7'],
    'l': ['1']
}

def obfuscate_word(word):
    return ''.join(
        random.choice(char_map[c]) if c in char_map and random.random() > 0.5 else c
        for c in word
    )

def obfuscate_text(text):
    return ' '.join(obfuscate_word(w) for w in text.split())
