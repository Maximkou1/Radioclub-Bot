from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

# line = input('enter anything')
# line_embedding = model.encode(line)
# line_embedding = np.float64(line_embedding) # приводим в одинаковый формат

# model = SentenceTransformer('cointegrated/rubert-tiny2')
# line = input('enter anything ')
# line_embedding = model.encode(line)
# line_embedding = np.float64(line_embedding) # приводим в одинаковый формат
# line_embedding = np.resize(line_embedding, 384)
# line_embedding = np.reshape(line_embedding, 384)
# print(line_embedding)

# model = SentenceTransformer("all-MiniLM-L6-v2")

model = SentenceTransformer('sentence-transformers/LaBSE')
df = pd.read_csv('lyrics_with_vectors.csv')
sentences = df['Lyrics'].tolist()

embeddings = df['Vectors'].apply(eval).apply(np.array)
embed_list = []
for i in range(len(embeddings)):
    embed_list.append(embeddings[i])
embeddings = np.array(embed_list)


def find_similars(line):
    line_embedding = model.encode(line)
    line_embedding = np.float64(line_embedding)
    cos_sim = util.cos_sim(line_embedding, embeddings).tolist()
    cos_sim = sum(cos_sim, [])

    all_sentence_combinations = []
    for i in range(len(cos_sim) - 1):
        all_sentence_combinations.append([cos_sim[i], sentences[i]])

    all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

    seen = set()
    uniq = []
    for score, sentence in all_sentence_combinations:
        if sentence not in seen:
            uniq.append((score, sentence))
            seen.add(sentence)

    # можно выводить по пять
    # similars = []
    # for score, sentence in uniq[0:5]:
    #     similars.append((sentence, score))

    # найдём просто самое популярное
    most_similar = uniq[0][1]

    return most_similar
