from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model = None

def load_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def tfidf_score(correct, student):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([correct, student])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return score

def sbert_score(correct, student):
    m = load_model()
    emb1 = m.encode(correct, convert_to_tensor=True)
    emb2 = m.encode(student, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def final_score(correct, student):
    t = tfidf_score(correct, student)
    s = sbert_score(correct, student)

    final = (0.2 * t + 0.8 * s) * 100
    return round(final, 2)
