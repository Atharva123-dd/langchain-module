from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "What is Python?": "Python is a popular programming language used for AI, web development, and automation.",

    "What is AI?": "AI (Artificial Intelligence) is the simulation of human intelligence in machines.",

    "What is Machine Learning?": "Machine Learning is a field of AI where systems learn from data without being explicitly programmed.",

    "What is Deep Learning?": "Deep Learning is a subset of ML that uses neural networks with many layers.",

    "What is LangChain?": "LangChain is a framework for building applications using large language models.",

    "What is NLP?": "Natural Language Processing helps computers understand human language.",

    "What is a Neural Network?": "A neural network is a computing system inspired by the human brain.",

    "What is Data Science?": "Data Science involves extracting insights from data using statistics and programming.",

    "What is a Model?": "A model is a trained system that makes predictions based on data.",

    "What is Training in AI?": "Training is the process where a model learns patterns from data."
}
questions = list(data.keys())
answers = list(data.values())

model = SentenceTransformer("BAAI/bge-small-en")

question_embeddings = model.encode(questions)

while True:

    user_question = input("\nAsk (or exit): ")

    if user_question == "exit":
        break

    user_embedding = model.encode([user_question])

    scores = cosine_similarity(user_embedding, question_embeddings)

    best_index = scores.argmax()
    score = scores[0][best_index]

    if score < 0.5:
        print("\nSorry, I don't know that.")
        continue

    print("\nAnswer:", answers[best_index])
    print("Confidence:", score)