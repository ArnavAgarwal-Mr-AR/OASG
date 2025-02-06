# 🎙️ Omniscient Audio Story Story Generator 

This project is a **Omniscient Audio Story Generator** that takes an input prompt in any Indian language, generates a story using a **fine-tuned LLaMA 3.1 model**, translates the output to the target language, and converts it into **speech using a Text-to-Speech (TTS) model**. The application runs on **Gradio** for an interactive interface.

---

## 🚀 Features
- **Prompt-based Story Generation** using a fine-tuned LLaMA 3.1 (8B) model.
- **Multi-language Support** (translates the generated story to a target language).
- **Text-to-Speech Conversion** (audio generation from text).
- **Supports Male & Female Voices** with tone customization.
- **Gradio UI for easy interaction**.

---

## 📦 Dependencies
Ensure you have the required dependencies installed:

```bash
pip install gradio transformers torch peft soundfile
pip install git+https://github.com/huggingface/parler-tts.git
```

For **4-bit quantization**:
```bash
pip install bitsandbytes
```

---

## 🔧 Setup Instructions

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/ArnavAgarwal-Mr-AR/OASG
cd OASG
```

### **2️⃣ Set Up Hugging Face Authentication**
Since **LLaMA 3.1 is a gated model**, authentication is required.

```bash
huggingface-cli login
```
Or, export your Hugging Face access token:
```bash
export HF_TOKEN="your-huggingface-token"
```

### **3️⃣ Run the Application**
```bash
pip install -r requirements.txt
python app.py
```

---

## ⚙️ Model Loading & Optimization
To prevent **memory issues**, the model is loaded with:
- **4-bit quantization (`load_in_4bit=True`)** to reduce memory usage.
- **CPU-only inference (`device_map={'': 'cpu'}`)** since GPUs are not available.
- **Fine-tuned adapter (`peft`)** for optimized story generation.


## 🎨 Gradio Interface
The Gradio interface allows users to input a prompt, select a language, and generate an audio story.

### **UI Components:**
- **Language Selection**: Choose input & target languages.
- **Gender & Tone Selection**: Pick male/female voice & tone.
- **Story Output**: Display generated text.
- **Audio Output**: Play or download the generated audio.

---

## 🛠 Troubleshooting
### **1️⃣ Memory Limit Exceeded (16GB RAM)**
- Try **using a smaller model** (e.g., `meta-llama/Meta-Llama-3-3B`).
- Use Hugging Face **Inference API** instead of local execution.

### **2️⃣ `bitsandbytes` Errors**
- Ensure it's installed:
  ```bash
  pip install bitsandbytes
  ```
- Disable if running on CPU:
  ```python
  os.environ["BITSANDBYTES_NOWELCOME"] = "1"
  ```

### **3️⃣ Model Authentication Errors**
- Log in to Hugging Face:
  ```bash
  huggingface-cli login
  ```
- Set `HF_TOKEN` manually if required.

---

## 🤝 Credits
- **LLaMA 3.1 Fine-Tuning** by `MrAR` (Me)
- **Translation Models** by `ai4bharat`
- **TTS Models** by `ai4bharat`
- **Gradio UI** for interaction

---

## 🌟 Support
For issues, raise an issue on the **GitHub repository** or contact us via Hugging Face.

## Contact me 📪
<div id="badges">
  <a href="https://www.linkedin.com/in/arnav-agarwal-571a59243/" target="blank">
   <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
 <a href="https://www.instagram.com/arnav_executes?igsh=MWUxaWlkanZob2lqeA==" target="blank">
 <img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"  alt="Instagram Badge" />
 </a>
 </a>
 <a href="https://medium.com/@arumynameis" target="blank">
 <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"  alt="Medium Badge" />
 </a>
</div>
