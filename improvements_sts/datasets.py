from torch.utils.data import Dataset
import torch
import random
import nltk
from nltk.corpus import wordnet
from tokenizer import BertTokenizer

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))

def synonym_replacement(sentence, n=1):
    words = nltk.word_tokenize(sentence)
    new_words = words.copy()
    random_word_list = [w for w in words if w.lower() not in STOPWORDS]
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            if synonym != random_word:
                new_words = [synonym if w == random_word else w for w in new_words]
                num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    return " ".join(new_words)

def random_deletion(sentence, p=0.1):
    words = nltk.word_tokenize(sentence)
    if len(words) == 1:  # nothing to delete
        return sentence
    new_words = [w for w in words if random.uniform(0, 1) > p]
    if not new_words:
        new_words = [random.choice(words)]
    return " ".join(new_words)

def random_swap(sentence, n=1):
    words = nltk.word_tokenize(sentence)
    for _ in range(n):
        if len(words) < 2:
            return sentence
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

def augment_sentence(sentence):
    """Apply a random augmentation strategy."""
    aug_choice = random.choice(["synonym", "deletion", "swap", "none"])
    if aug_choice == "synonym":
        return synonym_replacement(sentence, n=1)
    elif aug_choice == "deletion":
        return random_deletion(sentence, p=0.1)
    elif aug_choice == "swap":
        return random_swap(sentence, n=1)
    else:
        return sentence
    
    
class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False, augment_prob=0):
        self.dataset = dataset
        self.p = args
        self.augment_prob = augment_prob
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sent1, sent2, label, sent_id = self.dataset[idx]
        if self.augment_prob > 0:
            if random.random() < self.augment_prob:
                sent1 = augment_sentence(sent1)
            if random.random() < self.augment_prob:
                sent2 = augment_sentence(sent2)
            if random.random() < self.augment_prob:
                sent1, sent2 = sent2, sent1
        return (sent1, sent2, label, sent_id)

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors="pt", padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])
        if self.isRegression:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        ) = self.pad_data(all_data)

        batched_data = {
            "token_ids_1": token_ids,
            "token_type_ids_1": token_type_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "labels": labels,
            "sent_ids": sent_ids,
        }

        return batched_data