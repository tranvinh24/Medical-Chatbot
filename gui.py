import tkinter as tk
from tkinter import scrolledtext
import json
import random
import pickle
import torch
import numpy as np
from stem import preprocess_sentence
from nnModel import NeuralNet

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trợ lý Y tế Ảo")
        self.root.geometry("800x800")
        self.root.configure(bg="#f0f0f0")

        # Load model và data
        with open('data.pth', 'rb') as f:
            data = torch.load(f)

        self.model = NeuralNet(data["input_size"], data["hidden_size"], data["num_classes"])
        self.model.load_state_dict(data["model_state"])
        self.model.eval()

        with open('words.pkl', 'rb') as f:
            self.words = pickle.load(f)
        with open('classes.pkl', 'rb') as f:
            self.classes = pickle.load(f)

        # Tạo giao diện
        self.create_widgets()

    def create_widgets(self):
        # Frame chính
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Tiêu đề
        title_frame = tk.Frame(main_frame, bg="#4a90e2", height=60)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                             text=" Trợ lý Y tế Ảo ", 
                             font=("Arial", 22, "bold"),
                             bg="#4a90e2",
                             fg="white")
        title_label.pack(pady=10)

        # Khu vực chat
        chat_frame = tk.Frame(main_frame, bg="white", bd=2, relief=tk.SOLID)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chat_area = scrolledtext.ScrolledText(chat_frame,
                                                 wrap=tk.WORD,
                                                 width=50,
                                                 height=20,
                                                 font=("Arial", 16),
                                                 bg="white",
                                                 relief=tk.FLAT)
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)


        # Frame nhập liệu
        input_frame = tk.Frame(main_frame, bg="#f0f0f0")
        input_frame.pack(pady=10)

        # Ô nhập text
        self.msg_entry = tk.Entry(input_frame,
                                width=50,
                                font=("Arial", 16),
                                bd=2,
                                relief=tk.SOLID)
        self.msg_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        self.msg_entry.bind("<Return>", self.send_message)
        # Nút gửi
        send_button = tk.Button(input_frame,
                              text="Gửi",
                              command=self.send_message,
                              font=("Arial", 16, "bold"),
                              bg="#4a90e2",
                              fg="white",
                              relief=tk.FLAT,
                              padx=20,
                              cursor="hand2")
        send_button.pack(side=tk.RIGHT)

        # Thêm tin nhắn chào mừng
        self.add_bot_message("Xin chào! Tôi là Trợ lý Y tế Ảo. Tôi có thể giúp gì cho bạn?")

    def is_valid_question(self, sentence_words):
        # Kiểm tra xem có ít nhất một từ trong câu hỏi có trong từ điển không
        return any(word in self.words for word in sentence_words)

    def get_response(self, sentence, threshold=0.75):
        # Tiền xử lý câu
        sentence_words = preprocess_sentence(sentence)
        
        # Tạo bag of words
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1

        # Dự đoán
        X = torch.FloatTensor(bag)
        output = self.model(X)
        
        # Tính xác suất và độ tin cậy
        probabilities = torch.softmax(output, dim=0)
        confidence, predicted = torch.max(probabilities, dim=0)
        predicted_tag = self.classes[predicted.item()]

        # Nếu độ tin cậy thấp → trả lời không hiểu
        if confidence.item() < threshold:
            return "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể hỏi lại rõ ràng hơn không?"

        # Lấy câu trả lời ngẫu nhiên
        with open('intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)

        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['responses'])

        return "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể hỏi lại rõ ràng hơn không?"

    def add_bot_message(self, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, "🤖: " + message + "\n\n")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def add_user_message(self, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, "👤: " + message + "\n\n")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def send_message(self, event=None):
        message = self.msg_entry.get()
        if message.strip() != "":
            self.add_user_message(message)
            self.msg_entry.delete(0, tk.END)
            response = self.get_response(message)
            self.add_bot_message(response)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    # Cấu hình màu sắc cho tin nhắn
    app.chat_area.tag_configure("user_tag", foreground="black", font=("Arial", 16, "bold"))
    app.chat_area.tag_configure("user_message", foreground="black", font=("Arial", 16))
    app.chat_area.tag_configure("bot_tag", foreground="#4a90e2", font=("Arial", 16, "bold"))
    app.chat_area.tag_configure("bot_message", foreground="#4a90e2", font=("Arial", 16))
    root.mainloop() 