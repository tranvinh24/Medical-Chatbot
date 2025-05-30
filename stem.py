import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random

lemmatizer = WordNetLemmatizer()

def preprocess_sentence(sentence):
    # Chuyển về chữ thường và loại bỏ khoảng trắng thừa
    sentence = sentence.lower().strip()
    
    # Nếu câu chỉ có một từ, trả về list chứa từ đó
    if len(sentence.split()) == 1:
        return [sentence]
    
    # Tách từ tiếng Việt
    words = sentence.split()
    # Loại bỏ dấu câu
    words = [word for word in words if word.isalnum()]
    return words

def create_training_data():
    # Đọc file intents
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)

    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.', ',']

    # Xử lý từng intent
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tách từ
            w = preprocess_sentence(pattern)
            words.extend(w)
            # Thêm vào documents
            documents.append((w, intent['tag']))
            # Thêm vào classes
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Sắp xếp và loại bỏ trùng lặp
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique stemmed words", words)

    # Tạo training data
    training = []
    output_empty = [0] * len(classes)

    # Tạo bag of words cho mỗi document
    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    # Xáo trộn và chuyển thành numpy array
    random.shuffle(training)
    training = np.array(training, dtype=object)

    # Tách thành train_x và train_y
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    return train_x, train_y, words, classes 