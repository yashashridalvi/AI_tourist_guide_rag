# AI Tourist Guide Assistant (RAG Base)
An AI-powered multilingual tourist guide assistant that provides city travel information through voice or text conversation in English, Hindi, and Marathi.
It uses Retrieval Augmented Generation (RAG) with a structured city database and a local LLM (Ollama) to produce accurate, grounded, and natural responses.

### 📌 Problem Statement
Tourists often struggle to find reliable, structured, and language-friendly travel information such as transport, hotels, and sightseeing details.
Existing chatbots either:
1. Do not support Indian languages well
2. Provide hallucinated information
3. Do not support voice interaction

### Solution

This project builds a Siri-like AI Tourist Guide that:
1. Accepts voice or text input
2. Responds in English / Hindi / Marathi
3. Retrieves verified city data from a local database
4. Uses an LLM only to format and explain retrieved data
5. Speaks responses aloud in voice mode

Thus ensuring:
✅ Natural conversation
✅ Multilingual support
✅ Zero hallucination (because of RAG)
✅ Voice-based interaction

### Project Architecture 
User (Voice / Text)
        ↓
Speech-to-Text (if voice)
        ↓
Conversation Manager
        ↓
RAG Retriever (JSON City Database)
        ↓
LLM (Ollama TinyLlama / Phi-3)
        ↓
Response Generator
        ↓

Text-to-Speech (if voice)
        ↓
      User




      



