"""Experiments with scraped wikipedia data and NLP tools
"""
import json
import spacy
from pathlib import Path

# The path that WikiExtractor.py will extract to
JSONL_FILES_DIR = Path("text/AA")


def yield_all_articles(path: Path = JSONL_FILES_DIR):
    """Go through all the extracted wikipedia files and "yield" one at a time.

    Read about generators here if necessary: https://wiki.python.org/moin/Generators
    """
    for one_file in path.iterdir():
        print("Going through", one_file.name)
        with open(one_file) as f:
            for line in f:
                text = json.loads(line)["text"]
                yield text
                break


def count_things():

    article_gen = yield_all_articles(JSONL_FILES_DIR) #text to manipulate
    print("About to load spacy model")
    nlp = spacy.load("en")
    print("Finished loading spacy's English models")

    non_punct_non_space_token_count = 0
    lemma_count = 0
    ner_count_per_type = {}

    ###### Write below #########

    # 1. The total number of tokens that are not spaces or punctuations.
    # 2. The total number of lemmas.
    # 3. A count of the named entities to be found in the articles categorized by entity type.
    docs = list(nlp.pipe(article_gen))
    lem =[]

    for doc in docs:
        for token in doc:
            if (token.is_punct == False)or(token.is_space == False):
                non_punct_non_space_token_count += 1
                if token.lemma_ not in lem:
                    lem.append(token.lemma_)
        for ent in doc.ents:
            if ent.label_ not in ner_count_per_type.keys():
                ner_count_per_type[ent.label_] = 1
            else:
                ner_count_per_type[ent.label_] += 1
    lemma_count = len(lem)
    ###### End of your work #########

    print(
        "Non punctuation and non space token count: {}\nLemma count: {}".format(
            non_punct_non_space_token_count, lemma_count
        )
    )
    print("Named entity counts per type of named entity:")
    for ner_type, count in ner_count_per_type.items():
        print("{}: {}".format(ner_type, count))


if __name__ == "__main__":
    count_things()
