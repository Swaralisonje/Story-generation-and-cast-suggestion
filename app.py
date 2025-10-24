from flask import Flask, render_template, request, jsonify ,session ,  redirect, url_for , render_template_string
from llama_cpp import Llama
import os
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer, util
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import gender_guesser.detector as gender
import requests
from flask import send_file
from io import BytesIO
import sqlite3
import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId   # <-- needed for ObjectId conversions
from datetime import datetime, timedelta
import pytz

# ------------------------ Flask Setup ------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"


# ---------------------- MongoDB Setup ----------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["story_app"]
users = db["users"]
stories_collection = db["stories"]

# ------------------------ STORY GENERATION MODEL ------------------------
MODEL_PATH = r"C:\Users\Admin\PycharmProjects\MAJOR__PROJECT\model\llama-2-7b-chat.Q3_K_S.gguf"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

llama_model = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=8, n_gpu_layers=20, verbose=False)

# def get_db():
#     conn = sqlite3.connect("stories.db")
#     conn.row_factory = sqlite3.Row
#     return conn
#
# # Create table once (run this block once)
# def init_db():
#     conn = get_db()
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS stories (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             story TEXT,
#             genre TEXT,
#             keywords TEXT,
#             visibility TEXT,
#             timestamp TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()
#
# init_db()

def generate_story(keywords, genre):
    prompt = (
        f"Generate a Bollywood-style movie story in the {genre} genre using these keywords: {keywords}. "
        "The story should be 8–10 paragraphs, with main and side characters, emotional moments, "
        "and a strong climax. Write in continuous paragraphs, like a cinematic narration.\n\n"
        "After the story, suggest 2 songs name (do not include the singer's name) that suit the story and its situations. "
        "For each song, write a short 1-line abstract explaining the main meaning of the song or its lyrics. "

    )

    output = llama_model(
        prompt,
        max_tokens=10000,
        temperature=0.85,
        top_p=0.95,
        repeat_penalty=1.1
    )

    story = ""
    if isinstance(output, dict):
        if "choices" in output and len(output["choices"]) > 0:
            story = output["choices"][0].get("text", "").strip()
        else:
            story = output.get("text", "").strip()
    else:
        story = str(output).strip()

    paragraphs = [p.strip() for p in story.split("\n") if p.strip()]

    # Extract songs and abstracts from story text (you can parse them after a delimiter like "Songs:")
    # For simplicity, assume llama_model outputs songs at the end, separated by "---Songs---"
    if "---Songs---" in story:
        story_text, songs_text = story.split("---Songs---", 1)
    else:
        story_text = story
        songs_text = ""

    return paragraphs, story_text, songs_text


# ------------------------ CAST SUGGESTION MODEL ------------------------
ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load Actor Dataset
df = pd.read_csv("cast.csv", encoding="latin1")
df.columns = df.columns.str.strip().str.lower()
rename_map = {
    "actor name": "name",
    "type": "industry",
    "gender": "gender",
    "age": "age",
    "famous movies": "famous_movies",
    "famous genre": "famous_genre",
    "overall rating": "ratings"
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
df["ratings"] = pd.to_numeric(df["ratings"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["gender"] = df["gender"].str.lower()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for all actors
actor_texts = (df["famous_genre"].fillna("") + " " + df["famous_movies"].fillna("")).tolist()
actor_embeddings = embed_model.encode(actor_texts, convert_to_numpy=True)
dim = actor_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(actor_embeddings)

GENRES = {
    "Action": "Movies involving battles, fighting, war, soldiers, or heroes saving others.",
    "Romance": "Stories about love, relationships, emotions, or heartbreak.",
    "Comedy": "Lighthearted and funny moments, humor, jokes, or laughter.",
    "Thriller": "Suspenseful stories involving mystery, murder, or tension.",
    "Drama": "Emotional or serious human stories about life or struggles.",
    "Horror": "Supernatural or scary events involving ghosts or monsters.",
    "Sci-Fi": "Stories involving science, technology, or futuristic elements.",
    "Fantasy": "Mythical, magical, or unreal worlds and creatures."
}
genre_names = list(GENRES.keys())
genre_desc = list(GENRES.values())
genre_embeddings = embed_model.encode(genre_desc, convert_to_tensor=True)



# ------------------------ Helper Functions ------------------------
d = gender.Detector()
def detect_gender_api(name):
    if not name:
        return "Unknown"
    first_name = name.split()[0].capitalize()
    g = d.get_gender(first_name)
    if g in ["male", "mostly_male"]:
        return "Male"
    elif g in ["female", "mostly_female"]:
        return "Female"
    else:
        return "Unknown"

def extract_characters(story):
    entities = ner(story)
    chars = []
    seen = set()
    for e in entities:
        if e["entity_group"] == "PER":
            name = e["word"].strip()
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            pattern = r"([^.]*" + re.escape(name) + r"[^.]*)"
            m = re.search(pattern, story)
            sentence = m.group(0).strip() if m else name
            chars.append({"character": name, "sentence": sentence})
    return chars

def parse_age(sentence):
    m = re.search(r"(\d+)-year-old", sentence.lower())
    if m: return int(m.group(1))
    m = re.search(r"(\d+)\s*years?\s*old", sentence.lower())
    if m: return int(m.group(1))
    return None

def extract_genre_semantic(sentence):
    sentence_emb = embed_model.encode(sentence, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(sentence_emb, genre_embeddings)[0]
    best_idx = int(similarities.argmax())
    return genre_names[best_idx]

def suggest_actor(char):
    if char["predicted_gender"].lower() in ["male", "female"]:
        candidates = df[df["gender"].str.lower() == char["predicted_gender"].lower()].copy()
    else:
        candidates = df.copy()

    if char.get("genre_from_story"):
        genre_mask = candidates["famous_genre"].fillna("").str.contains(char["genre_from_story"], case=False)
        candidates = candidates[genre_mask]
        if candidates.empty and char["predicted_gender"].lower() in ["male", "female"]:
            candidates = df[df["gender"].str.lower() == char["predicted_gender"].lower()].copy()
    if candidates.empty:
        candidates = df.copy()

    actor_texts_filtered = (
        candidates["famous_genre"].fillna("") + " " + candidates["famous_movies"].fillna("")
    ).tolist()

    filtered_embeddings = embed_model.encode(actor_texts_filtered, convert_to_numpy=True)
    role_emb = embed_model.encode([char["sentence"]], convert_to_numpy=True)
    sims = cosine_similarity(role_emb, filtered_embeddings)[0]
    candidates = candidates.reset_index(drop=True)
    candidates["sim_score"] = sims

    if char["age"] is not None:
        candidates["age_score"] = 1 - (abs(candidates["age"] - char["age"]) / 100)
    else:
        candidates["age_score"] = 0.5

    candidates["rating_score"] = candidates["ratings"] / 10
    candidates["total_score"] = (
        0.5 * candidates["sim_score"] + 0.2 * candidates["age_score"] + 0.3 * candidates["rating_score"]
    )

    top = candidates.sort_values(by="total_score", ascending=False).head(3)

    return [
        {
            "actor_name": row["name"],
            "actor_genre": row["famous_genre"],
            "actor_rating": row["ratings"],
            "total_score": round(row["total_score"], 2),
        }
        for _, row in top.iterrows()
    ]


# ---------------------- Auth HTML ----------------------
register_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Register | StoryGen</title>
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: #550000;
            color: #ffffff;
            text-align: center;
            padding: 0;
            margin: 0; /* Add this line */
        }

        nav {
            background: #fff8f9; /* Changed */
            padding: 12px 0;
            box-shadow: 0 2px 10px rgba(155, 34, 38, 0.2); /* Changed */
        }

        nav a {
            color: #9B2226; /* Changed */
            margin: 0 15px;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }

        nav a:hover {
            color: #c84347; /* Changed */
        }

        .center-box {
            background: #fff8f9; /* Changed */
            border: 1px solid #f2dfe2; /* Changed */
            border-radius: 16px;
            width: 850px;
            margin: 120px auto;
            padding: 80px 25px;
            box-shadow: 0px 4px 25px rgba(155, 34, 38, 0.15); /* Changed */
        }

        h2 {
            color: #9B2226; /* Changed */
            margin-bottom: 25px;
            font-weight: 800;
            text-shadow: 0 0 8px rgba(155, 34, 38, 0.2); /* Changed */
        }

        input {
            width: 80%;
            padding: 20px;
            margin: 20px 0;
            border-radius: 12px;
            border: 1px solid #ddd; /* Changed */
            outline: none;
            background: #ffffff; /* Changed */
            color: #333; /* Changed */
        }

        input::placeholder {
            color: #aaa; /* Changed */
        }

        button {
            width: 85%;
            padding: 20px;
            margin-top: 20px;
            background: #9B2226; /* Changed */
            color: #fff;
            border: none;
            border-radius: 10px;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 0 10px rgba(155, 34, 38, 0.3); /* Changed */
        }

        button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(155, 34, 38, 0.6); /* Changed */
        }

        p {
            color: #ae2012; /* Changed */
            font-weight: 500;
        }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('login') }}">Login</a>
        <a href="{{ url_for('register') }}">Register</a>
    </nav>

    <div class="center-box">
        <h2>📝 Register</h2>
        <form method="POST">
            <input type="text" name="name" placeholder="Full Name" required><br>
            <input type="email" name="email" placeholder="Email" required><br>
            <input type="password" name="password" placeholder="Password" required><br>
            <button type="submit">Create Account</button>
        </form>
        {% if message %}
        <p>{{ message }}</p>
        {% endif %}
    </div>
</body>
</html>
"""

login_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Login | StoryGen</title>
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: #550000;
            color: #ffffff;
            text-align: center;
            padding: 0;
            margin: 0; /* Add this line */
        }

        nav {
            background: #fff8f9; /* Changed */
            padding: 18px 0;
            box-shadow: 0 2px 10px rgba(155, 34, 38, 0.2); /* Changed */
        }

        nav a {
            color: #9B2226; /* Changed */
            margin: 0 15px;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }

        nav a:hover {
            color: #c84347; /* Changed */
        }

        .center-box {
            background: #fff8f9; /* Changed */
            border: 1px solid #f2dfe2; /* Changed */
            border-radius: 16px;
            width: 850px;
            margin: 120px auto;
            padding: 80px 25px;
            box-shadow: 0px 4px 25px rgba(155, 34, 38, 0.15); /* Changed */
        }

        h2 {
            color: #9B2226; /* Changed */
            margin-bottom: 25px;
            font-weight: 600;
            text-shadow: 0 0 8px rgba(155, 34, 38, 0.2); /* Changed */
        }

        input {
            width: 80%;
            padding: 20px;
            margin: 20px 0;
            border-radius: 12px;
            border: 1px solid #ddd; /* Changed */
            outline: none;
            background: #ffffff; /* Changed */
            color: #333; /* Changed */
        }

        input::placeholder {
            color: #aaa; /* Changed */
        }

        button {
            width: 85%;
            padding: 20px;
            margin-top: 20px;
            background: #9B2226; /* Changed */
            color: #fff;
            border: none;
            border-radius: 10px;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 0 10px rgba(155, 34, 38, 0.3); /* Changed */
        }

        button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(155, 34, 38, 0.6); /* Changed */
        }

        p {
            color: #ae2012; /* Changed */
            font-weight: 500;
        }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('login') }}">Login</a>
        <a href="{{ url_for('register') }}">Register</a>
    </nav>

    <div class="center-box">

        <h2>🎥 Login</h2>

        <form method="POST">
            <input type="email" name="email" placeholder="Email" required><br>
            <input type="password" name="password" placeholder="Password" required><br>
            <button type="submit">Login</button>
        </form>
        {% if message %}
        <p>{{ message }}</p>
        {% endif %}
    </div>
</body>
</html>
"""


# ------------------------ ROUTES ------------------------
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    message = None
    if request.method == "POST":
        n = request.form["name"]
        e = request.form["email"]
        p = request.form["password"]
        if users.find_one({"email": e}):
            message = "User already exists!"
        else:
            users.insert_one({"name": n, "email": e, "password": p})
            return redirect(url_for("login"))
    return render_template_string(register_html, message=message)

@app.route("/login", methods=["GET", "POST"])
def login():
    message = None
    if request.method == "POST":
        e = request.form["email"]
        p = request.form["password"]
        u = users.find_one({"email": e})
        if not u:
            message = "User not found."
        elif u["password"] != p:
            message = "Wrong password."
        else:
            session["user"] = e
            return redirect(url_for("index"))
    return render_template_string(login_html, message=message)

@app.route("/index", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        keywords = request.form.get("keywords")
        genre = request.form.get("genre")
        paragraphs, full_story, songs_and_abstracts = generate_story(keywords, genre)
        return render_template("story.html", story_paragraphs=paragraphs, full_story=full_story)
    return render_template("index.html")


@app.route("/get_cast", methods=["POST"])
def get_cast():
    story = request.form.get("story", "")
    results = []
    if story:
        characters = extract_characters(story)
        for c in characters:
            c["age"] = parse_age(c["sentence"])
            c["predicted_gender"] = detect_gender_api(c["character"])
            c["genre_from_story"] = extract_genre_semantic(c["sentence"])
            c["suggestions"] = suggest_actor(c)
            results.append(c)
    return jsonify(results)

def get_ist_time():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")


@app.route("/save_story", methods=["POST"])
def save_story():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        story = data.get("story", "").strip()
        title = data.get("title", "Untitled Story")
        visibility = data.get("visibility")  # public/private
        user_email = session.get("user")

        if not user_email:
            return jsonify({"success": False, "redirect": url_for("login")}), 401
        if not story:
            return jsonify({"success": False, "error": "Story is empty"}), 400

        # 🇮🇳 Get current Indian time
        ist = pytz.timezone("Asia/Kolkata")
        timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

        story_data = {
            "user_email": user_email,
            "title": title,
            "story": story,
            "visibility": visibility,
            "timestamp": timestamp
        }

        stories_collection.insert_one(story_data)
        return jsonify({"success": True}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500





@app.route("/feed")
def feed():
    # Show all public stories, newest first
    public_stories = list(
        stories_collection.find({"visibility": "public"}).sort("timestamp", -1)
    )
    return render_template("feed.html", stories=public_stories)


@app.route("/history")
def history():
    user_email = session.get("user")
    if not user_email:
        return redirect(url_for("login"))

    stories = list(stories_collection.find({"user_email": user_email}).sort("timestamp", -1))
    return render_template("history.html", stories=stories)



@app.route("/remix_story", methods=["POST"])
def remix_story():
    try:
        data = request.get_json()
        story_id = data.get("story_id")
        old_story = data.get("story", "")
        user_prompt = data.get("prompt", "")

        if not user_prompt:
            return jsonify({"success": False, "error": "Missing remix prompt"}), 400

        if not story_id:
            return jsonify({"success": False, "error": "Missing story id"}), 400

        # ✅ Generate new story using your model
        remix_input = (
            f"Rewrite the following story creatively based on this idea: '{user_prompt}'.\n\n"
            f"Original Story:\n{old_story}\n\n"
            "Write the remixed story in 5 to 6 engaging paragraphs, keeping the same characters but changing the plot and tone to make it feel like a Bollywood-style story. "
            "Ensure smooth flow, emotional depth, and a strong narrative. "
            "At the end, suggest 1 suitable song name for this new version (do not include the singer's name) and write a short 1-line abstract explaining the meaning or essence of the song."
        )
        result = llama_model(remix_input, max_tokens=7000, temperature=0.7, top_p=0.9)
        new_story = result["choices"][0]["text"].strip()

        # ✅ Save remixed story in MongoDB
        user_email = session.get("user")
        remix_doc = {
            "user_email": user_email,
            "story": new_story,
            "original_story_id": ObjectId(story_id),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "visibility": "private",  # or public if you want
        }
        inserted = stories_collection.insert_one(remix_doc)

        # ✅ Return the URL to redirect to view_story page
        return jsonify({
            "success": True,
            "message": "Your story generated successfully!",
            "redirect_url": f"/view_story/{inserted.inserted_id}"
        })

    except Exception as e:
        print("❌ Remix error:", str(e))
        return jsonify({"success": False, "error": str(e)})

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

# @app.route("/store_story", methods=["POST"])
# def store_story():
#     data = request.get_json()
#     story = data.get("story")
#     title = data.get("title")
#     visibility = data.get("visibility")  # 'public' or 'private'
#
#     if not story or not title:
#         return jsonify({"error": "Missing story or title"}), 400
#
#     new_entry = {
#         "title": title,
#         "story": story,
#         "visibility": visibility
#     }

    # # Save to History (always)
    # history_path = "data/history.csv"
    # import pandas as pd
    # try:
    #     df = pd.read_csv(history_path)
    # except FileNotFoundError:
    #     df = pd.DataFrame(columns=["title", "story", "visibility"])
    #
    # df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    # df.to_csv(history_path, index=False)
    #
    # # If public, also store in feed
    # if visibility == "public":
    #     feed_path = "data/feed.csv"
    #     try:
    #         df_feed = pd.read_csv(feed_path)
    #     except FileNotFoundError:
    #         df_feed = pd.DataFrame(columns=["title", "story"])
    #     df_feed = pd.concat([df_feed, pd.DataFrame([{"title": title, "story": story}])], ignore_index=True)
    #     df_feed.to_csv(feed_path, index=False)
    #
    # return jsonify({"success": True, "message": f"Story stored {visibility}ly"})


@app.route("/view_story/<story_id>")
def view_story(story_id):
    try:
        story = stories_collection.find_one({"_id": ObjectId(story_id)})
        if not story:
            return "Story not found", 404

        # Fetch all remixes of this story
        remixes = list(stories_collection.find({"original_story_id": ObjectId(story_id)}).sort("timestamp", 1))

        return render_template("view_story.html", story=story, remixes=remixes)

    except Exception as e:
        print("Error loading story:", str(e))
        return "Server Error", 500


@app.route("/delete_story", methods=["POST"])
def delete_story():
    user_email = session.get("user")
    if not user_email:
        return jsonify({"success": False, "redirect": url_for("login")}), 401

    data = request.get_json()
    if not data or "id" not in data:
        return jsonify({"success": False, "error": "Missing story id"}), 400

    story_id = data["id"]
    try:
        obj_id = ObjectId(story_id)
    except Exception:
        return jsonify({"success": False, "error": "Invalid story id"}), 400

    result = stories_collection.delete_one({"_id": obj_id, "user_email": user_email})

    if result.deleted_count == 1:
        return "", 204  # successfully deleted
    else:
        return jsonify({"success": False, "error": "Story not found or not owned"}), 404


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ------------------------ Run App ------------------------
import colorama
colorama.deinit()
if __name__ == "__main__":
    app.run(debug=True)
# llama_model = Llama(model_path="model/llama-2-7b-chat.Q4_K_M.gguf")