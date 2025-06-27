import gradio as gr
from tts_rvc_logic import process_tts_and_rvc, get_available_models
from methods.topic_sentiment_summarization import MultitaskPipeline, category_labels, sentiment_labels, positive_words, negative_words
from transformers import pipeline
import os
import json


def analyze_and_display(dataset_path):
    if not dataset_path:
        return "No dataset uploaded.", "No dataset uploaded.", None, None

    # Pass the required arguments to the pipeline
    multitask_pipeline = MultitaskPipeline(
        topic_model_path="./models/bert-topic-classification-turkish", 
        zero_shot_model_name="./local_models/xlm-roberta-large-xnli", 
        tokenizer_path="./models/bert-topic-classification-turkish", 
        category_labels=category_labels, 
        sentiment_labels=sentiment_labels, 
        positive_words=positive_words, 
        negative_words=negative_words
    )

    multitask_path, summarized_path = analyze_dataset(dataset_path.name, multitask_pipeline)

    with open(multitask_path, "r", encoding="utf-8") as f:
        multitask_content = json.dumps(json.load(f), ensure_ascii=False, indent=4)

    with open(summarized_path, "r", encoding="utf-8") as f:
        summarized_content = json.dumps(json.load(f), ensure_ascii=False, indent=4)

    return multitask_content, summarized_content, multitask_path, summarized_path


# Function to convert numbers to Turkish words
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

def get_models_in_subfolders(base_folder="rvc_models"):
    models = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".pth"):
                models.append(os.path.relpath(os.path.join(root, file), base_folder))
    return models

def analyze_dataset(dataset_path, pipeline):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter customer messages
    customer_texts = [item for item in data if item['speaker'] == 'customer']

    # Combine messages by conversation_id
    import pandas as pd
    df = pd.DataFrame(data)
    df['combined_text'] = df.groupby('conversation_id')['text'].transform(lambda x: ' '.join(x))
    df = df.drop_duplicates(subset='conversation_id')

    results = []

    for conversation_id in df['conversation_id'].unique():
        combined_text = df[df['conversation_id'] == conversation_id]['combined_text'].iloc[0]
        customer_text_list = [item['text'] for item in customer_texts if item['conversation_id'] == conversation_id]

        # Analyze topic
        topic = pipeline.analyze_topic(combined_text)

        # Analyze sentiment for customer messages
        sentiment_results = []
        for customer_text in customer_text_list:
            sentiment, confidence = pipeline.analyze_sentiment(customer_text)
            sentiment_results.append({
                "conversation_id": int(conversation_id),
                "text": customer_text,
                "sentiment": sentiment,
                "confidence": float(confidence)
            })

        results.append({
            "conversation_id": int(conversation_id),
            "combined_text": combined_text,
            "topic": topic,
            "sentiments": sentiment_results
        })

    # Save multitask output
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    multitask_output_path = os.path.join(output_dir, "multitask_output.json")
    with open(multitask_output_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    # Summarize results
    summarized_results = []

    for result in results:
        conversation_id = result['conversation_id']
        conversation_id_words = number_to_words_turkish(conversation_id)
        topic = result['topic']
        sentiments = result['sentiments']

        summary = f"Aydisi {conversation_id_words} olan telefon konuşması, '{topic}' konusu üzerine gerçekleşmiştir. "
        positive_count = sum(1 for sentiment in sentiments if sentiment['sentiment'] == "Pozitif")
        negative_count = sum(1 for sentiment in sentiments if sentiment['sentiment'] == "Negatif")
        if positive_count > negative_count:
            summary += "Müşteri konuşması genelde pozitif bir duygusal ton taşımaktadır."
        elif negative_count > positive_count:
            summary += "Müşteri konuşması genelde negatif bir duygusal ton taşımaktadır."
        else:
            summary += "Müşteri konuşması nötr bir duygusal ton taşımaktadır."


        avg_confidence = sum(sentiment['confidence'] for sentiment in sentiments) / len(sentiments)

        summarized_results.append({
            "conversation_id": conversation_id,
            "summary": summary,
            "average_confidence": avg_confidence
        })

    summarized_output_path = os.path.join(output_dir, "summarized_output.json")
    with open(summarized_output_path, "w", encoding="utf-8") as json_file:
        json.dump(summarized_results, json_file, ensure_ascii=False, indent=4)

    return multitask_output_path, summarized_output_path

def interface():
    available_models = get_models_in_subfolders()

    if not available_models:
        available_models = ["No models available"]

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", radius_size="lg")) as demo:
        gr.Markdown("## Dataset Analysis and TTS to RVC Voice Conversion")

        with gr.Tab("Dataset Analysis"):
            with gr.Row():
                dataset_input = gr.File(label="Upload Dataset (JSON)", file_types=[".json"])
                with gr.Row():
                    multitask_download = gr.File(label="Download Multitask Output")
                    summarized_download = gr.File(label="Download Summarized Output")

            analyze_button = gr.Button("Analyze Dataset")
            
            with gr.Row():
                multitask_output_box = gr.Textbox(label="Multitask Output Preview", lines=35, interactive=False)
                summarized_output_box = gr.Textbox(label="Summarized Output Preview", lines=35, interactive=False)
  
            analyze_button.click(
                analyze_and_display,
                inputs=[dataset_input],
                outputs=[multitask_output_box, summarized_output_box, multitask_download, summarized_download]
            )


        with gr.Tab("Voice Conversion"):
            with gr.Row():
                json_input = gr.File(label="Upload JSON for TTS", file_types=[".json"])
                with gr.Row():
                    with gr.Column(scale=1):
                        rvc_model_dropdown = gr.Dropdown(label="Choose RVC Voice Model", choices=available_models)
                        pitch_slider = gr.Slider(label="Pitch Adjustment", minimum=-24, maximum=24, step=1, value=0)
                    rvc_model_input = gr.File(label="Or Upload RVC Voice Model (.pth)", file_types=[".pth"])

            generate_button = gr.Button("Generate Voice")
            audio_output = gr.Audio(label="Converted Voice Output")

            def tts_and_convert(json_file, selected_model, uploaded_model, pitch):
                with open(json_file.name, "r", encoding="utf-8") as f:
                    data = json.load(f)

                summaries = [item['summary'] for item in data]
                combined_text = " ".join(summaries)

                rvc_model_path = uploaded_model.name if uploaded_model else os.path.join("rvc_models", selected_model)
                output_path = process_tts_and_rvc(combined_text, rvc_model_path, pitch)
                return output_path

            generate_button.click(
                tts_and_convert,
                inputs=[json_input, rvc_model_dropdown, rvc_model_input, pitch_slider],
                outputs=audio_output
            )

    return demo

if __name__ == "__main__":
    demo = interface()
    demo.launch()
