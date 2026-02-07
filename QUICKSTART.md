# Quick Start Guide - PDF Q&A Assistant

## 🚀 Get Running in 5 Minutes

### Step 1: Install Python
Make sure you have Python 3.8+ installed:
```bash
python --version
```

### Step 2: Clone/Download the Project
```bash
# If using git
git clone https://github.com/dev-k99/RAG-Assistant
cd pdf-rag-app

# OR simply download the files to a folder
```

### Step 3: Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**Note**: First install will download ~90MB of embedding models. This is normal!

### Step 4: Get Your Free Groq API Key

1. Go to https://console.groq.com
2. Click "Sign Up" (it's free!)
3. After signing in, go to "API Keys"
4. Click "Create API Key"
5. Copy the key (starts with "gsk_...")

### Step 5: Run the App
```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

### Step 6: Use the App

1. **Paste your Groq API key** in the sidebar
2. **Upload a PDF** file
3. **Click "Process PDF"** (wait ~5-10 seconds)
4. **Start asking questions!**

---

## 📦 What Gets Installed?

```
streamlit          → Web interface
langchain          → RAG orchestration
langchain-groq     → Free LLM API integration
faiss-cpu          → Vector search database
pypdf              → PDF text extraction
sentence-transformers → Local embeddings (no API needed)
```

**Total size**: ~500MB after installation

---

## 🎯 Example Questions to Try

Once you've uploaded a PDF, try these:

- "What is this document about?"
- "Summarize the main points"
- "What are the key findings in section 2?"
- "List all the recommendations"
- "Explain [specific term] from the document"

---

## 🐛 Troubleshooting

**"No module named 'streamlit'"**
→ Make sure you activated the virtual environment and ran `pip install -r requirements.txt`

**"Could not extract text from PDF"**
→ Your PDF might be scanned/image-based. Try a different PDF with selectable text.

**App is slow on first run**
→ Normal! It's downloading the embedding model. Next runs will be fast.

**"API key invalid"**
→ Double-check you copied the entire key from Groq console.

---

## 💡 Pro Tips

1. **Better answers**: Ask specific questions referencing sections or topics
2. **Multiple questions**: The chat remembers context, so you can ask follow-ups
3. **Clear chat**: Re-process the PDF to start fresh
4. **Large PDFs**: May take longer to process (up to 50MB supported)

---

## 📞 Need Help?

- Check the main README.md for detailed documentation
- Create an issue on GitHub
- Review the code - it's well-commented!

---

**Ready to go? Run `streamlit run app.py` and start querying your PDFs! 🎉**