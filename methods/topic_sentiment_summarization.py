from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from transformers import pipeline as transformers_pipeline
import torch
import pandas as pd
import json
import os

from transformers import BertForSequenceClassification, BertTokenizer, pipeline
import torch

class MultitaskPipeline:
    def __init__(self, topic_model_path, zero_shot_model_name, tokenizer_path, category_labels, sentiment_labels,
                 positive_words, negative_words):
        # Load topic classification model
        self.topic_model = BertForSequenceClassification.from_pretrained(topic_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.category_labels = category_labels

        # Load zero-shot sentiment classification model
        self.zero_shot_classifier = transformers_pipeline("zero-shot-classification", model=zero_shot_model_name, device=0)
        self.sentiment_labels = sentiment_labels

        # Positive and negative word lists
        self.positive_words = positive_words
        self.negative_words = negative_words

    def analyze_topic(self, combined_text):
        # Perform topic classification on full conversation
        inputs = self.tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            topic_logits = self.topic_model(**inputs).logits
            topic_label_id = torch.argmax(topic_logits, dim=1).item()
            topic_label = self.category_labels[topic_label_id]
        return topic_label

    def analyze_sentiment(self, customer_text):
        # Perform zero-shot sentiment classification
        sentiment_result = self.zero_shot_classifier(customer_text, self.sentiment_labels, multi_label=False)
        sentiment_label = sentiment_result['labels'][0]
        sentiment_confidence = sentiment_result['scores'][0]

        # If confidence is low, fall back to keyword-based analysis
        if sentiment_confidence < 0.5:
            words = customer_text.lower().split()
            pos_count = sum(1 for word in words if word in self.positive_words)
            neg_count = sum(1 for word in words if word in self.negative_words)

            if pos_count > neg_count:
                sentiment_label = "Pozitif"
            elif neg_count > pos_count:
                sentiment_label = "Negatif"

        return sentiment_label, sentiment_confidence

# Topic categories and sentiment classes
category_labels = [
    "Finansal Hizmetler",
    "Hesap İşlemleri",
    "Teknik Destek",
    "Ürün ve Satış",
    "İade ve Değişim"
]
sentiment_labels = ["Pozitif", "Negatif"]

# Positive and negative word lists
positive_words = [
    "iyi", "güzel", "harika", "mükemmel", "pozitif", "başarılı", "mutlu",
    "memnun", "sevgi", "çok teşekkürler", "süper", "Anlayışlı", "Başarılı",
    "Esnek", "Etkili", "Güler yüzlü", "Güvenilir", "Kaliteli", "Kolay",
    "Olumlu", "Profesyonel", "Sevinç", "Tatmin edici", "Tavsiye",
    "Tecrübeli", "Teşekkür", "Yardımcı", "Yardımsever", "Zamanında",
    "Şahane", "Çözüm", "Çalışkan", "İlgili"
]
negative_words = [
    "kötü", "berbat", "felaket", "hata", "şikayet", "olumsuz", "sorun", "sorunum var",
    "üzgün", "zarar", "hayal kırıklığı", "pişmanım", "sinir bozucu",
    "Başarısız", "Beklenmedik", "Eksik", "Gereksiz", "Gergin", "Hatalı",
    "Kapanmış", "Kayıp", "Kusurlu", "Kırılmış", "Mağdur", "Memnuniyetsiz",
    "Problemli", "Rahatsız", "Sinir bozucu", "Sorunlu", "Stres",
    "Uğraştırıcı", "Yanlış", "Yavaş", "Yoğunluk", "Zaman kaybı", "Zorlayıcı",
    "Çözülmemiş", "İlgisiz", "İstemiyorum"
]

# Create multitask pipeline instance
pipeline = MultitaskPipeline(
    topic_model_path="./models/bert-topic-classification-turkish",
    zero_shot_model_name="./local_models/xlm-roberta-large-xnli",
    tokenizer_path="./models/bert-topic-classification-turkish",
    category_labels=category_labels,
    sentiment_labels=sentiment_labels,
    positive_words=positive_words,
    negative_words=negative_words
)

# Load dataset
file_path = './datasets/test_dataset.json'
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Select only customer utterances
customer_texts = [item for item in data if item['speaker'] == 'customer']

# Merge texts by conversation ID
df = pd.DataFrame(data)
df['combined_text'] = df.groupby('conversation_id')['text'].transform(lambda x: ' '.join(x))
df = df.drop_duplicates(subset='conversation_id')

# Store analysis results
results = []

for conversation_id in df['conversation_id'].unique():
    combined_text = df[df['conversation_id'] == conversation_id]['combined_text'].iloc[0]
    customer_text_list = [item['text'] for item in customer_texts if item['conversation_id'] == conversation_id]

    # Perform topic analysis
    topic = pipeline.analyze_topic(combined_text)

    # Perform sentiment analysis for each customer sentence
    sentiment_results = []
    for customer_text in customer_text_list:
        sentiment, confidence = pipeline.analyze_sentiment(customer_text)
        sentiment_results.append({
            "conversation_id": int(conversation_id),
            "text": customer_text,
            "sentiment": sentiment,
            "confidence": float(confidence)
        })

    # Store result per conversation
    results.append({
        "conversation_id": int(conversation_id),
        "combined_text": combined_text,
        "topic": topic,
        "sentiments": sentiment_results
    })

# Save results as JSON
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "multitask_output.json"), "w", encoding="utf-8") as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

print("Results saved: outputs/multitask_output.json")

# Function to convert numeric values to Turkish words
def number_to_words_turkish(number):
    ones = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
    tens = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
    hundreds = ["", "yüz", "ikiyüz", "üçyüz", "dörtyüz", "beşyüz", "altıyüz", "yediyüz", "sekizyüz", "dokuzyüz"]

    if number == 0:
        return "sıfır"

    if number < 0:
        return "eksi " + number_to_words_turkish(abs(number))

    words = ""

    if number >= 1000:
        thousands = number // 1000
        if thousands == 1:  # Avoid saying "bir bin"
            words += "bin "
        else:
            words += ones[thousands] + " bin "
        number %= 1000

    if number >= 100:
        hundreds_digit = number // 100
        words += hundreds[hundreds_digit] + " "
        number %= 100

    if number >= 10:
        tens_digit = number // 10
        words += tens[tens_digit] + " "
        number %= 10

    if number > 0:
        words += ones[number]

    return words.strip()

# Generate summaries and compute average confidence
summarized_results = []

for result in results:
    conversation_id = result['conversation_id']
    topic = result['topic']
    sentiments = result['sentiments']

    # Convert numeric ID to Turkish words
    conversation_id_words = number_to_words_turkish(int(conversation_id))
    
    # Create summary text
    summary = f"Phone call with ID {conversation_id_words} was about '{topic}'. "
    positive_count = sum(1 for sentiment in sentiments if sentiment['sentiment'] == "Pozitif")
    negative_count = sum(1 for sentiment in sentiments if sentiment['sentiment'] == "Negatif")
    if positive_count > negative_count:
        summary += "Customer sentiment was mostly positive."
    elif negative_count > positive_count:
        summary += "Customer sentiment was mostly negative."
    else:
        summary += "Customer sentiment was neutral."

    # Calculate average confidence
    avg_confidence = sum(sentiment['confidence'] for sentiment in sentiments) / len(sentiments)

    # Store final summary
    summarized_results.append({
        "conversation_id": conversation_id,
        "summary": summary,
        "average_confidence": avg_confidence
    })

# Save summarized output
with open(os.path.join(output_dir, "summarized_output.json"), "w", encoding="utf-8") as json_file:
    json.dump(summarized_results, json_file, ensure_ascii=False, indent=4)

print("Summarized results saved: outputs/summarized_output.json")
