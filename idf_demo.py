"""Quick demo of IDF keyword selection."""

import argparse
import collections
import itertools

import spacy

from lm_eval import tasks


OUTLINE = '%-64s %s'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--spacy_model', type=str, default='en_core_web_sm')
    parser.add_argument('--omit_answers', action='store_true')
    parser.add_argument('--include_stopwords', action='store_true')
    parser.add_argument('--lowercase', action='store_true')
    parser.add_argument('--k', type=int, default=5)
    return parser.parse_args()


def main():
    """Prints out low-frequency terms for each instance."""
    args = parse_args()

    task = tasks.get_task(args.task)()
    nlp = spacy.load(args.spacy_model)

    print(OUTLINE % ('IDF Terms', 'Prompt'))

    docs = []
    if task.has_training_docs():
        docs.extend(task.training_docs())
    if task.has_validation_docs():
        docs.extend(task.validation_docs())
    if task.has_test_docs():
        docs.extend(task.test_docs())

    def _to_text(doc):
        text = task.doc_to_text(doc)
        if not args.omit_answers:
            text += task.doc_to_target(doc)
        return text

    def _extract_terms(doc):
        if not args.include_stopwords:
            doc = filter(lambda x: not x.is_stop, doc)
        terms = [token.text for token in doc]
        if args.lowercase:
            terms = map(lambda x: x.lower(), terms)
        return list(terms)

    texts = [_to_text(doc) for doc in docs]
    processed_texts = nlp.pipe(texts, batch_size=1024, n_process=16,
                               disable=['parser', 'tagger', 'ner', 'lemmatizer'])
    terms = [_extract_terms(doc) for doc in processed_texts]
    term_freqs = collections.Counter(itertools.chain(*terms))

    for text, term in zip(texts, terms):
        freq = [term_freqs[x] for x in term]
        out = sorted(zip(term, freq), key=lambda x: x[1])
        prefix = ', '.join(x[0] for x in out[:args.k])
        print(OUTLINE % (prefix, text.replace('\n', ' ')))


if __name__ == '__main__':
    main()
