import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from spellchecker import SpellChecker
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import load_model
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords as nltk_stopwords
from transformers import BertTokenizer, BertModel
import torch
import os
import json
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

stopwords_set = set(nltk_stopwords.words('english'))

model = load_model('./SavedModels/essay_score_model.h5')

wordvec_model = Word2Vec.load('./SavedModels/wordvec_model', mmap=None)

stdscaler = StandardScaler()

additional_features = pd.read_pickle('./SavedModels/training_features.pkl')

references = {}

feature_list = [
    'word_count',
    'corrections',
    'similarity',
    'token_count',
    'unique_token_count',
    'nostop_count',
    'sent_count',
    'ner_count',
    'comma',
    'question',
    'exclamation',
    'quotation',
    'organization',
    'caps',
    'person',
    'location',
    'money',
    'time',
    'date',
    'percent',
    'noun',
    'adj',
    'pron',
    'verb',
    'cconj',
    'adv',
    'det',
    'propn',
    'num',
    'part',
    'intj'
]

stdscaler.fit(additional_features[feature_list])
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(additional_features[feature_list])

# Function for cleaning and preprocessing text
def preprocess_text(text):
    nlp = spacy.load('en_core_web_sm')
    spell = SpellChecker()
    corrected_text = ' '.join(spell.correction(word) if word in spell else word for word in text.split())

    # Clean text (remove personal pronouns, stopwords, and punctuation)
    tokens = [tok.lemma_.lower().strip() for tok in nlp(corrected_text, disable=['parser', 'ner'])  
              if tok.lemma_ != '-PRON-' and tok.text.lower() not in stopwords_set and tok.text not in string.punctuation]
    
    cleaned_text = ' '.join(tokens)
    word_vectors = [wordvec_model.wv[word] for word in cleaned_text.split() if word in wordvec_model.wv.key_to_index]
    if word_vectors:
        average_vec = np.mean(word_vectors, axis=0)
    else:
        average_vec = np.zeros((300,), dtype='float32')  

    return cleaned_text, average_vec

# Function to extract additional features
def extract_features(text):
    nlp = spacy.load('en_core_web_sm')  
    spell = SpellChecker()  

    features = {
        'word_count': len(text.split()),
        'corrections': len(spell.unknown(spell.split_words(text))), 
        'similarity': references.get('similarity', 2),  
        'token_count': len(list(nlp(text, disable=['parser', 'ner']))),  
        'unique_token_count': len(set(list(nlp(text, disable=['parser', 'ner'])))),
        'nostop_count': len([token for token in list(nlp(text, disable=['parser', 'ner']))
                            if token.text.lower() not in stopwords_set]),
        'sent_count': len(list(nlp(text).sents)), 
        'ner_count': len(list(nlp(text).ents)), 
        'comma': text.count(','),
        'question': text.count('?'),
        'exclamation': text.count('!'),
        'quotation': text.count('"') + text.count("'"),
        'organization': text.count(r'@ORGANIZATION'),
        'caps': text.count(r'@CAPS'),
        'person': text.count(r'@PERSON'),
        'location': text.count(r'@LOCATION'),
        'money': text.count(r'@MONEY'),
        'time': text.count(r'@TIME'),
        'date': text.count(r'@DATE'),
        'percent': text.count(r'@PERCENT'),
        'noun': text.count('NOUN'),
        'adj': text.count('ADJ'),
        'pron': text.count('PRON'),
        'verb': text.count('VERB'),
        'cconj': text.count('CCONJ'),
        'adv': text.count('ADV'),
        'det': text.count('DET'),
        'propn': text.count('PROPN'),
        'num': text.count('NUM'),
        'part': text.count('PART'),
        'intj': text.count('INTJ')
    }
    return features


def predict_score(essay_text):
    cleaned_text, word_vector = preprocess_text(essay_text)
    additional_features = extract_features(cleaned_text)
    features = stdscaler.transform([[additional_features[feature] for feature in feature_list]])
    flattened_word_vector = np.array(word_vector).flatten()
    features = features.reshape(1, -1)                               # Reshape features and word vector to ensure they're 2D arrays
    flattened_word_vector = flattened_word_vector.reshape(1, -1)
    combined_features = np.concatenate([features, flattened_word_vector], axis=1)
    prediction = model.predict(combined_features)
    prediction = prediction.reshape(1, -1)
    score = round(prediction.item())
    return score


def calculate_similarity_score(essay_content, reference_essay):
    vectorizer = TfidfVectorizer()
    essay_vectors = vectorizer.fit_transform([essay_content, reference_essay])
    similarity_matrix = cosine_similarity(essay_vectors)
    similarity_score = similarity_matrix[0, 1]

    return similarity_score


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def calculate_bert_sentence_similarity(sentence1, sentence2):

    tokens_sentence1 = tokenizer.encode(sentence1, add_special_tokens=True)
    tokens_sentence2 = tokenizer.encode(sentence2, add_special_tokens=True)
    input_ids_sentence1 = torch.tensor(tokens_sentence1).unsqueeze(0)
    input_ids_sentence2 = torch.tensor(tokens_sentence2).unsqueeze(0)
    with torch.no_grad():
        outputs_sentence1 = bert_model(input_ids_sentence1)
        outputs_sentence2 = bert_model(input_ids_sentence2)
    cls_embedding_sentence1 = outputs_sentence1.last_hidden_state[:, 0, :]
    cls_embedding_sentence2 = outputs_sentence2.last_hidden_state[:, 0, :]
    similarity = torch.nn.functional.cosine_similarity(cls_embedding_sentence1, cls_embedding_sentence2, dim=1)

    return similarity.item()


def calculate_coherence_score(essay_content):
    sentences = essay_content.split('.')
    coherence_penalty = 0.3                   # Decrease in creativity score for each non-related sentence
    topic_sentence = sentences[0]             # Assuming the first sentence represents the overall topic
    coherence_score = 1.0                     # Initialize coherence score
    for sentence in sentences[1:]:
        similarity = calculate_bert_sentence_similarity(topic_sentence, sentence)
        coherence_score -= coherence_penalty * (1.0 - similarity)

    return max(coherence_score, 0.0)


def assess_creativity_and_similarity(essay_content, reference_essay):
    diverse_vocabulary_score = calculate_diverse_vocabulary_score(essay_content)
    originality_score = calculate_originality_score(essay_content)
    imaginative_elements_score = calculate_imaginative_elements_score(essay_content)
    coherence_score = calculate_coherence_score(essay_content)

    # Adjust weights as needed
    overall_creativity_score = 0.3 * diverse_vocabulary_score + 0.2 * originality_score + 0.4 * imaginative_elements_score - 1.0 * coherence_score
    similarity_score = calculate_similarity_score(essay_content, reference_essay)
    scaled_similarity_score = max(0, min(5, (similarity_score)))

    return overall_creativity_score, scaled_similarity_score


def calculate_diverse_vocabulary_score(essay_content):
    unique_words = set(word.lower() for word in essay_content.split())
    return min(len(unique_words) / 50, 1.0)


def calculate_originality_score(essay_content):
    phrases = essay_content.split('.')
    unique_phrases = set(phrase.strip().lower() for phrase in phrases)
    return min(len(unique_phrases) / 10, 1.0)


def calculate_imaginative_elements_score(essay_content):
    imaginative_keywords = ['imagine', 'create', 'innovate', 'fantasy', 'dream']
    imaginative_count = sum(essay_content.lower().count(keyword) for keyword in imaginative_keywords)
    return min(imaginative_count / 5, 1.0)

