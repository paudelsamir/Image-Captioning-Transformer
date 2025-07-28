import re
from collections import Counter
import pickle
import os

def clean_caption(caption):
    """Clean and preprocess caption text"""
    caption = caption.lower()
    caption = caption.strip()
    # remove punctuations (except periods for sentence structure)
    caption = re.sub(r'[^\w\s.]', '', caption)
    # remove extra spaces
    caption = re.sub(r'\s+', ' ', caption)
    # remove numbers/digits
    caption = re.sub(r'\d+', '', caption)

    # add start and end tokens
    caption = '<start> ' + caption + ' <end>'
    return caption

def build_vocabulary(captions, min_word_freq=2):
    """Build vocabulary from captions"""
    # Count all words
    word_counts = Counter()
    
    for caption in captions:
        words = caption.split()
        word_counts.update(words)
    
    print(f"Total unique words found: {len(word_counts)}")
    print(f"Most common words: {word_counts.most_common(10)}")
    
    # Filter words by frequency but EXCLUDE special tokens from filtered list
    special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
    vocab_words = [word for word, count in word_counts.items() 
                   if count >= min_word_freq and word not in special_tokens]
    
    # Add special tokens at the beginning
    vocab = special_tokens + vocab_words
    
    print(f"Vocabulary size (min_freq={min_word_freq}): {len(vocab)}")
    
    # Create mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return vocab, word2idx, idx2word, word_counts

def tokenize_caption(caption, word2idx, max_length=50):
    """Convert caption to sequence of token indices"""
    words = caption.split()
    tokens = []
    unk_count = 0
    
    # Convert words to indices
    for word in words:
        if word in word2idx:
            tokens.append(word2idx[word])
        else:
            tokens.append(word2idx['<unk>'])
            unk_count += 1
    
    # Pad or truncate to max_length
    if len(tokens) < max_length:
        tokens.extend([word2idx['<pad>']] * (max_length - len(tokens)))
    else:
        tokens = tokens[:max_length]
    
    return tokens, unk_count

def load_vocabulary(vocab_path='vocabulary.pkl'):
    """Load vocabulary from pickle file"""
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        return vocab_data['vocab'], vocab_data['word2idx'], vocab_data['idx2word']
    else:
        # Default vocabulary if file doesn't exist
        vocab = ['<pad>', '<unk>', '<start>', '<end>', 'a', 'dog', 'cat', 'person', 'sitting', 'standing']
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        return vocab, word2idx, idx2word
