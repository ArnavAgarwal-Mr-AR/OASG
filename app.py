import gradio as gr
import torch
import subprocess
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoModelForCausalLM
from IndicTransToolkit import IndicProcessor
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import numpy as np
from accelerate import infer_auto_device_map
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Initialize IndicTransToolkit
indic_processor = IndicProcessor(inference=True)

# Load translation models
indic_to_en_model = "ai4bharat/indictrans2-indic-en-1B"
en_to_indic_model = "ai4bharat/indictrans2-en-indic-1B"

indic_to_en_tokenizer = AutoTokenizer.from_pretrained(indic_to_en_model, trust_remote_code=True)
indic_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(indic_to_en_model, trust_remote_code=True).to(DEVICE)

en_to_indic_tokenizer = AutoTokenizer.from_pretrained(en_to_indic_model, trust_remote_code=True)
en_to_indic_model = AutoModelForSeq2SeqLM.from_pretrained(en_to_indic_model, trust_remote_code=True).to(DEVICE)

# Hugging Face access token
HF_TOKEN = os.getenv("HF_TOKEN")  

# Load LLaMA 3.1 base model using Hugging Face token
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
story_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
# Try to load in 4-bit mode with CPU offloading
story_model = PeftModel.from_pretrained(base_model, "MrAR/NarrativeStoryGen").to(DEVICE)



# Load fine-tuned adapter model
story_model = PeftModel.from_pretrained(base_model, "MrAR/NarrativeStoryGen").to(DEVICE)



# Load TTS model (Parler-TTS)
tts_model_name = "ai4bharat/indic-parler-tts"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_name).to(DEVICE)
tts_processor = AutoTokenizer.from_pretrained(tts_model_name)
description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)

# Language mappings for IndicTransToolkit
LANG_CODES = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Dogri": "doi_Deva",
    "English": "eng_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Manipuri": "mni_Beng",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu"
}

# Recommended speakers
SPEAKER_TONES = {
    "Assamese": {"Male": "Amit", "Female": "Sita"},
    "Bengali": {"Male": "Arjun", "Female": "Aditi"},
    "Bodo": {"Male": "Bikram", "Female": "Maya"},
    "Chhattisgarhi": {"Male": "Bhanu", "Female": "Champa"},
    "Dogri": {"Male": "Karan", "Female": "Karan"},
    "English": {"Male": "Thoma", "Female": "Mary"},
    "Gujarati": {"Male": "Yash", "Female": "Neha"},
    "Hindi": {"Male": "Rohit", "Female": "Divya"},
    "Kannada": {"Male": "Suresh", "Female": "Anu"},
    "Malayalam": {"Male": "Harish", "Female": "Anjali"},
    "Manipuri": {"Male": "Laishram", "Female": "Ranjit"},
    "Marathi": {"Male": "Sanjay", "Female": "Sunita"},
    "Nepali": {"Male": "Amrita", "Female": "Amrita"},
    "Odia": {"Male": "Manas", "Female": "Debjani"},
    "Punjabi": {"Male": "Divjot", "Female": "Gurpreet"},
    "Sanskrit": {"Male": "Aryan", "Female": "Aryan"},
    "Tamil": {"Male": "Jaya", "Female": "Jaya"},
    "Telugu": {"Male": "Prakash", "Female": "Lalitha"}
}

def story_generator(prompt):
    inputs = story_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = story_model.generate(**inputs, max_length=1500)
    return story_tokenizer.decode(output[0], skip_special_tokens=True)

# Translation function using IndicTransToolkit
def translate_text(text, src_lang, tgt_lang, tokenizer, model):
    batch = indic_processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return indic_processor.postprocess_batch([translated_text], lang=tgt_lang)[0]
    
# TTS function using Parler-TTS
def text_to_speech(text, language, gender, tone):
    # Determine the speaker based on language and gender
    speaker_info = SPEAKER_TONES.get(language, {}).get(gender)
    speaker_name = speaker_info["name"]
    
    # Automatically create a description including the speaker's voice tone
    description = f"{speaker_name}'s voice is {tone} with no background noise. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
    
    description_input_ids = description_tokenizer(description, return_tensors="pt").to(DEVICE)
    prompt_input_ids = tts_processor(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generation = tts_model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask
        )
    
    audio_arr = generation.cpu().numpy().squeeze()
    output_audio_path = "output_story.wav"
    sf.write(output_audio_path, audio_arr, tts_model.config.sampling_rate)

    return output_audio_path