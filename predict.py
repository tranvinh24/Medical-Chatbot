import torch
import json
import random
import pickle
import numpy as np
from stem import preprocess_sentence
from nnModel import NeuralNet

def load_model():
    # Load model và data
    with open('data.pth', 'rb') as f:
        data = torch.load(f)

    # Khởi tạo model
    model = NeuralNet(data["input_size"], data["hidden_size"], data["num_classes"])
    model.load_state_dict(data["model_state"])
    model.eval()

    # Load words và classes
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)

    return model, words, classes

def is_valid_question(sentence_words, words):
    # Kiểm tra xem có ít nhất một từ trong câu hỏi có trong từ điển không
    return any(word in words for word in sentence_words)

def get_response(sentence, model, words, classes, threshold=0.75):
    # Tiền xử lý câu
    sentence_words = preprocess_sentence(sentence)

    # Tạo bag of words
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    # Dự đoán
    X = torch.FloatTensor(bag)
    output = model(X)
    
    # Tính xác suất và độ tin cậy
    probabilities = torch.softmax(output, dim=0)
    confidence, predicted = torch.max(probabilities, dim=0)
    predicted_tag = classes[predicted.item()]

    # Load intents
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

    # Nếu độ tin cậy thấp → trả lời không hiểu
    if confidence.item() < threshold:
        return "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể hỏi lại rõ ràng hơn không?"

    # Nếu độ tin cậy cao → trả về câu trả lời tương ứng
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

    # Trường hợp không khớp tag nào
    return "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể hỏi lại rõ ràng hơn không?"

def chat():
    print("Bắt đầu trò chuyện! (gõ 'quit' để thoát)")
    model, words, classes = load_model()
    
    while True:
        sentence = input("Bạn: ")
        if sentence.lower() == 'quit':
            break

        response = get_response(sentence, model, words, classes)
        print("Bot:", response)

if __name__ == "__main__":
    chat() 