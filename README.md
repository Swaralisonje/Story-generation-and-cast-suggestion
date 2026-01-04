**🎬 PlotBot — AI Story Generation & Cast Suggestion Platform**

**🧠 Overview**

PlotBot is an AI-powered story generation web application built using Flask.
It allows users to generate movie-like stories based on keywords and genres using a LLaMA-2 model, then automatically suggests the perfect cast using NLP models.

Users can:

Generate cinematic stories

Get intelligent cast suggestions

Remix or rewrite stories

Store stories privately or publicly

View personal story history or explore the global feed


## 🚀 Features

| Feature                | Description                                                                                                               |
|------------------------|---------------------------------------------------------------------------------------------------------------------------|
| 🧩 Story Generation    | Generates stories using LLaMA 2 (local model).                                                                            |
| 🎭 Cast Suggestion     | Suggests suitable actors based on extracted character traits using BERT NER + SentenceTransformer similarity.            |
| 🔐 User Authentication | Login & Registration using Flask sessions and MongoDB.                                                                    |
| 💾 Story Storage       | Save stories as Private (History) or Public (Feed) with timestamps in IST.                                                |
| 🔁 Thread Feature      | Modify or rewrite existing stories with a short user prompt.                                                              |
| 🧱 MongoDB Integration | Secure, persistent data storage for users and stories.                                                                    |


## 🛠️ Tech Stack

| Category   | Technology                                                                           |
|-----------|--------------------------------------------------------------------------------------|
| Frontend  | HTML, CSS (Bootstrap 5), JavaScript, jQuery                                         |
| Backend   | Flask (Python)                                                                      |
| AI Models | LLaMA-2-7B (local), BERT (for NER), SentenceTransformer (`all-MiniLM-L6-v2`)        |
| Database  | MongoDB                                                                             |
| Libraries | `transformers`, `sentence-transformers`, `faiss`, `pandas`, `pytz`, `gender-guesser`|



**📚 How It Works**

🧠 1. Story Generation

Input keywords and genre → Flask sends prompt to LLaMA-2 model

Model returns a cinematic story with optional song suggestions

🎭 2. Cast Suggestion

NER identifies characters from the story

SentenceTransformer encodes character context and matches with cast.csv

Suggests top 3 actors based on semantic match, age, gender, and rating

💾 3. Story Storage

Public → Appears in global Feed

Private → Saved in My Stories

Latest stories appear at the top with IST timestamp

🔁 4. Thread Creation

Rewrite existing stories with a short prompt (e.g., “Make it a tragic ending”)

Saves as a new story in MongoDB


**🧠 AI Models Used**

| Model               | Purpose                                                |
|--------------------|--------------------------------------------------------|
| LLaMA-2-7B          | Generates Bollywood-style stories                      |
| BERT (NER)          | Extracts character names and entities from story text |
| SentenceTransformer | Encodes semantic meaning for cast matching            |



**🙏 Acknowledgements**

Inspired by NLP-based story generation research and open-source AI models

Thanks to LLaMA, BERT, and SentenceTransformer teams for their contributions
