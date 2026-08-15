from collections import Counter
import re

corpus = """
Large language models are advanced artificial intelligence systems designed to understand, generate, and interact with human language. These models are typically built using deep learning architectures, particularly transformers, and are trained on vast amounts of text data scraped from the internet. They can perform a wide range of tasks, including translation, summarization, question answering, and even creative writing. The fundamental mechanism involves predicting the next word in a sequence, allowing them to construct coherent and contextually relevant responses. As research progresses, these models are becoming increasingly sophisticated, demonstrating capabilities that mimic human-like reasoning and comprehension, though they still lack true understanding and are prone to generating hallucinations or biased content based on their training data.
"""

def build_vocab(text):
    """
    Creates a frequency dictionary from the input text.
    Each word is split into characters separated by spaces, with an end-of-word symbol '_'.
    """
    word_counts = {}

    for word in text.split():
        key = " ".join(list(word)) + " _"
        word_counts[key] = word_counts.get(key, 0) + 1

    return word_counts

def get_pairs_from_word(word):
    """
    Takes a formatted word string and extracts adjacent symbol pairs (bigrams).
    """
    symbols = word.split()
    return list(zip(symbols, symbols[1:]))


def get_all_pairs_from_vocab(word_counts):
    """
    Iterates over the vocabulary and extracts symbol pairs for each word.
    The pairs are repeated based on the word's frequency in the corpus.
    """
    all_pairs = []
    for keys, values in word_counts.items():
        for _ in range(values):              # ← repeat by frequency!
            all_pairs.append(get_pairs_from_word(keys))
    return all_pairs

def count_pair_frequencies(all_pairs):
    """
    Flattens the nested list of pairs and counts the occurrences of each unique pair.
    """
    all_counter = []
    count = 0
    for word in all_pairs:
        for pair in word:
            all_counter.append(pair)


    return Counter(all_counter)

def merger(word_counts, all_count_pair_frequencies):
    top_pair, count = all_count_pair_frequencies.most_common(1)[0]
    spaced_pair = " ".join(top_pair)  
    merged_pair = "".join(top_pair)

    new_dic = {}

    for keys, values in word_counts.items():
        # Just replace the spaced version with the merged version directly!
        new_keys = keys.replace(spaced_pair, merged_pair)
        new_dic[new_keys] = values
    return merged_pair, count, new_dic
    

def train(vocab, num_merges):
    merges = []

    for i in range(num_merges):
        all_pairs = get_all_pairs_from_vocab(vocab)
        ranked = count_pair_frequencies(all_pairs)
        merged_pair, count, vocab = merger(vocab, ranked)
        merges.append(merged_pair)
        # print(f"Merge {i+1}:")
        # print(f"Most frequent pair merged: '{merged_pair}' (occurred {count} times)")
        # print(f"Updated Vocabulary:\n{vocab}\n{'-'*50}")
    return vocab, merges    


def encode(text, merges):
    """
    Tokenize new text using the learned BPE merges (applied in order).
    """
    words = text.split()
    all_token_ids = []

    for word in words:
        # Start with individual characters + end-of-word symbol
        symbols = list(word) + ["_"]
        
        # Apply each merge in the order they were learned
        for merged in merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                # Try to match the merged token starting at position i
                if i < len(symbols) - 1 and symbols[i] + symbols[i+1] == merged:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        
        all_token_ids.extend(symbols)
    
    return all_token_ids


def decode(token_ids, id_to_token):
    tokens = [id_to_token[i] for i in token_ids]
    text = "".join(tokens)
    text = text.replace("_", " ")
    return text.strip()

def build_token_ids(merges, corpus):
    """
    Build token vocabulary from all unique characters + learned merges.
    """
    # Base characters (unique chars in corpus)
    base_chars = sorted(set(c for c in corpus if not c.isspace()))
    base_chars.append("_")  # end-of-word symbol
    
    all_tokens = base_chars + merges  # order: chars first, then merged tokens
    
    token_to_id = {token: idx for idx, token in enumerate(all_tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token

vocab = build_vocab(corpus)
print(f"Step 1 - build_vocab:\n{vocab}\n{'-'*50}")

merged_keys = get_all_pairs_from_vocab(vocab)
# print(f"Step 2 - get_all_pairs_from_vocab:\n{merged_keys}\n{'-'*50}")

pairs = count_pair_frequencies(merged_keys)
print(f"Step 3 - count_pair_frequencies:\n{pairs}\n{'-'*50}")

merged_pair, count, final_dict = merger(vocab, pairs)
print(f"Step 4 - merger:")
print(f"Most frequent pair merged: '{merged_pair}' (occurred {count} times)")
print(f"Updated Vocabulary:\n{final_dict}\n{'-'*50}")


vocab, merges = train(vocab, 200)
print(f"Final Vocabulary:\n{vocab}")
print(f"Merges:\n{merges}")

test_sentence = "LLM is based on attention mechanism"

token_to_id, id_to_token = build_token_ids(merges, corpus)

test_sentence = "Shubham Prakash is great."
tokens = encode(test_sentence, merges)
print("Tokens:", tokens)

# Convert tokens to IDs (handle unknown tokens gracefully)
token_ids = [token_to_id.get(t, token_to_id.get("_", 0)) for t in tokens]
print("Token IDs:", token_ids)

decoded = decode(token_ids, id_to_token)
print("Decoded:", decoded)