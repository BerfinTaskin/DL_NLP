import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from torch.utils.data import Dataset
import torch
import random
from tokenizer import BertTokenizer
from wordfreq import zipf_frequency
import inflect
p = inflect.engine()
STOPWORDS = set(stopwords.words("english"))

def get_wordnet_pos(word):
    """Map POS tag from nltk.pos_tag to WordNet POS."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def filter_synonyms(base, synonyms, threshold=2.5):
    """Filter synonyms by frequency, blacklist rules, and exclude weird ones."""
    filtered = []
    for s in synonyms:
        if (len(s) <= 15
            and s.isalpha()
            and zipf_frequency(s, "en") >= threshold):
            filtered.append(s)
    return filtered

def match_morphology(original, synonym):
    """Try to match plural/verb tense form of the original word."""
    # Plural handling
    if original.endswith("s") and not synonym.endswith("s"):
        return p.plural(synonym)
    if not original.endswith("s") and synonym.endswith("s"):
        return p.singular_noun(synonym) or synonym
    # Verb tense (very rough)
    if original.endswith("ing") and not synonym.endswith("ing"):
        return synonym + "ing"
    if original.endswith("ed") and not synonym.endswith("ed"):
        return synonym + "ed"
    return synonym

def synonym_replacement(sentence, n):
    """
    Replace up to n non-stopwords in the sentence with a random synonym
    that matches the word's POS, filters out multi-word phrases and numbers,
    and applies frequency/morphology/capitalization improvements.
    """
    words = nltk.word_tokenize(sentence)
    new_words = words.copy()
    random_word_list = [w for w in words if w.lower() not in STOPWORDS]
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        pos = get_wordnet_pos(random_word)
        base = lemmatizer.lemmatize(random_word.lower(), pos=pos)

        synonyms = set()
        for syn in wordnet.synsets(base, pos=pos)[:2]:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != base:
                    synonyms.add(synonym)

        # apply filters
        synonyms = filter_synonyms(base, synonyms)
        if synonyms:
            synonym = random.choice(synonyms)
            synonym = match_morphology(random_word, synonym)

            # capitalization
            if random_word[0].isupper():
                synonym = synonym.capitalize()

            for i, w in enumerate(new_words):
                if w == random_word:
                    new_words[i] = synonym
                    num_replaced += 1
                    break
        if num_replaced >= n:
            break
    return " ".join(new_words)

def random_deletion(sentence, p):
    """
    Randomly delete each word in the sentence with probability p.
    If all words are removed, keep one random word.
    This simulates missing or noisy tokens.
    """
    words = nltk.word_tokenize(sentence)
    if len(words) == 1:  # nothing to delete
        return sentence
    new_words = [w for w in words if random.uniform(0, 1) > p]
    if not new_words:
        new_words = [random.choice(words)]
    return " ".join(new_words)

def random_swap(sentence, n):
    """
    Randomly swap the positions of two words in the sentence, repeated n times.
    Introduces small word-order perturbations while retaining content.
    """
    words = nltk.word_tokenize(sentence)
    for _ in range(n):
        if len(words) < 2:
            return sentence
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

def augment_sentence(sentence, augment_prob):
    if random.random() < augment_prob:
        sentence = synonym_replacement(sentence, n=1)
    if random.random() < augment_prob:
        sentence = random_deletion(sentence, p=0.1)
    if random.random() < augment_prob:
        sentence = random_swap(sentence, n=1)
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
        sent1 = augment_sentence(sent1, self.augment_prob)
        sent2 = augment_sentence(sent2, self.augment_prob)
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