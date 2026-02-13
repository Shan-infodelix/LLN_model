from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("./local_sbert_model")

def tfidf_score(correct, student):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([correct, student])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return score

def sbert_score(correct, student):
    emb1 = model.encode(correct, convert_to_tensor=True)
    emb2 = model.encode(student, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def final_score(correct, student):
    t = tfidf_score(correct, student)
    s = sbert_score(correct, student)

    final = (0.2 * t + 0.8 * s) * 100
    return round(final, 2)
