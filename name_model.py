import string
from random import randint, choices
from collections import defaultdict
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# -----------------------
# Configuration
# -----------------------

DATA_PATH = "data.csv"   # CSV with columns: string,label
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------
# Constants
# -----------------------

ALPHABET = string.ascii_lowercase
CHAR_TO_INT = {c: i for i, c in enumerate(ALPHABET)}
VOWELS = set("aeiou")

consonants = "TNSRHDLCMFWGPBVKXQJZ".lower()
consonants_w = [
    0.1519498722,
    0.1160304504,
    0.1048909409,
    0.1005578916,
    0.09889062944,
    0.07213198853,
    0.06644314361,
    0.04528173982,
    0.0436144777,
    0.03847527963,
    0.03498502212,
    0.03383076373,
    0.03037714935,
    0.02487152005,
    0.01849561657,
    0.01151510155,
    0.002885645972,
    0.001877960077,
    0.001722226803,
    0.001172579951
]

vow = "aeiouy"
vowels_w = [
    0.2024828416,
    0.2995816357,
    0.182084165,
    0.1914495092,
    0.07172349676,
    0.0526783517
]


def random_name():
    syl = []
    for i in range(3):
        if randint(0, 1):
            syl.append("".join(choices(consonants, weights=consonants_w)))
            syl.append("".join(choices(vow, weights=vowels_w)))
        else:
            syl.append("".join(choices(vow, weights=vowels_w)))
            syl.append("".join(choices(consonants, weights=consonants_w)))

    return "".join(syl)




class nameGeneratorModel:
    def __init__(self, dataset:str):
        self.dataset = dataset

    def extract_features(self, word):
        features = []

        # 1. One-hot per character position
        for pos, ch in enumerate(word):
            one_hot = [0] * 26
            one_hot[CHAR_TO_INT[ch]] = 1
            features.extend(one_hot)

        # 2. Vowel flags
        for ch in word:
            features.append(1 if ch in VOWELS else 0)

        # 3. V/C transitions
        for i in range(5):
            a = 1 if word[i] in VOWELS else 0
            b = 1 if word[i+1] in VOWELS else 0
            features.append(a * 2 + b)

        # 4. Global features
        features.append(sum(ch in VOWELS for ch in word))
        features.append(len(set(word)))
        features.append(int(any(word[i] == word[i+1] for i in range(5))))

        bad_bigram_count = 0
        for i in range(5):
            if word[i:i+2] in self.BAD_BIGRAMS:
                bad_bigram_count += 1

        features.append(bad_bigram_count)

        return features

    def get_feature_names(self, word_len=6):
        names = []

        # 1. One-hot per character position
        for pos in range(word_len):
            for c in ALPHABET:
                names.append(f"pos{pos}_{c}")

        # 2. Vowel flags
        for pos in range(word_len):
            names.append(f"is_vowel_{pos}")

        # 3. V/C transitions
        for i in range(word_len - 1):
            names.append(f"vc_transition_{i}")

        # 4. Global features
        names += [
            "num_vowels",
            "unique_chars",
            "has_double_char",
            "bad_bigram_count",
        ]

        return names

    def load_data(self):
        self.df = pd.read_csv(self.dataset)

        bigram_stats = defaultdict(lambda: [0, 0])  # [accepted, total]

        for word, label in zip(self.df["string"], self.df["label"]):
            for i in range(5):
                bg = word[i:i+2]
                bigram_stats[bg][1] += 1
                if label == 1:
                    bigram_stats[bg][0] += 1

        self.BAD_BIGRAMS = {
            bg for bg, (acc, tot) in bigram_stats.items()
            if tot >= 5 and acc / tot < 0.2
        }

        feature_names = self.get_feature_names()
        X_array = [self.extract_features(w) for w in self.df["string"]]
        self.X = pd.DataFrame(X_array, columns=feature_names)
        self.y = self.df["label"].values

    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=TEST_SIZE,
            stratify=self.y,
            random_state=RANDOM_STATE
        )

        self.model = LGBMClassifier(
            verbosity=-1,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE
        )

        CATEGORICAL_FEATURES = list(range(6))
        self.model.fit(self.X_train, self.y_train, categorical_feature=CATEGORICAL_FEATURES)

    def test(self):
        y_pred = self.model.predict(self.X_test)

        print("\nAccuracy:", accuracy_score(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

    def predict_string(self, word):
        """
        Return acceptance probability for a new string.
        """

        features = pd.DataFrame(
        [self.extract_features(word)],
        columns=self.X.columns
        )
        prob = self.model.predict_proba(features)[0, 1]
        return prob

    def predict_batch(self, words):
        X = np.array([self.extract_features(w) for w in words])
        return self.model.predict_proba(X)[:, 1]

    def generate_names(self, n=20, threshold=0.7, max_attempts=100_000):
        names = set()
        nonNames = set()
        attempts = 0

        while len(names) < n and attempts < max_attempts:
            attempts += 1
            candidate = random_name()
            prob = self.predict_string(candidate)

            if prob >= threshold:
                names.add(candidate)

        return sorted(names)

    def retrain_model(self):
        self.load_data()
        self.train()
        self.test()
        print(self.df["label"].value_counts(normalize=True))
        probs = self.model.predict_proba(self.X_test)[:, 1]
        print("ROC AUC:", roc_auc_score(self.y_test, probs))


if __name__ == "__main__":
    genModel = nameGeneratorModel("data.csv")
    genModel.retrain_model()

    generated = genModel.generate_names()

    for name in generated:
        print(f"{name}, 1")