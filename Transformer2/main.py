import csv
import numpy as np
from transformer import Transformer
import spacy
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    sample = 2000
    nlp = spacy.load("en_core_web_sm")
    path = "D:/datasets/sampuran/"

    english = list()
    with open(path+"english2.csv", encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for row in reader:
            english.append(row[0])

    en_corpus = list()
    for j in range(sample):
        doc = nlp(english[j])
        text = list()
        for i in range(len(doc)):
            text.append(str(doc[i]))
        en_corpus.append(text)

    UNK = '<UNK>'
    PAD = '<PAD>'
    BOS = '<BOS>'
    EOS = '<EOS>'

    en2id = {BOS: 0, EOS: 1, UNK: 2, PAD: 3}
    id2en = {}
    id2en[0] = BOS
    id2en[1] = EOS
    id2en[2] = UNK
    id2en[3] = PAD

    for j in range(sample):
        for i in range(len(en_corpus[j])):
            word = en_corpus[j][i]
            if word not in en2id:
                new_id = len(en2id)
                en2id[word] = new_id
                id2en[new_id] = word

    for i in range(sample):
        en_corpus[i].append(EOS)
        en_corpus[i].insert(0, BOS)
    print(en_corpus[0])

    nlp = spacy.load("ja_core_news_sm")

    japanese = list()
    with open(path+"japanese2.csv", encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for row in reader:
            japanese.append(row[0])
    # print(len(sentences2))

    jp_corpus = list()
    for j in range(sample):
        doc = nlp(japanese[j])
        text = list()
        for i in range(len(doc)):
            text.append(str(doc[i]))
        jp_corpus.append(text)

    jp2id = {BOS: 0, EOS: 1, UNK: 2, PAD: 3}

    id2jp = {}
    id2jp[0] = BOS
    id2jp[1] = EOS
    id2jp[2] = UNK
    id2jp[3] = PAD

    for j in range(sample):
        for i in range(len(jp_corpus[j])):
            word = jp_corpus[j][i]
            if word not in jp2id:
                new_id = len(jp2id)
                jp2id[word] = new_id
                id2jp[new_id] = word

    for i in range(sample):
        jp_corpus[i].append(EOS)
        jp_corpus[i].insert(0, BOS)

    epochs = 1
    batch_size = 16
    block = int(sample / batch_size)
    loss = np.zeros(epochs)

    en_vocab_size = len(id2en)
    jp_vocab_size = len(id2jp)
    print(en_vocab_size)
    print(jp_vocab_size)

    tf = Transformer(jp_vocab_size)

    lr = 1e-5
    for t in range(epochs):
        start = time.time()
        for k in range(block):
            print(k)
            en_texts = en_corpus[k*batch_size:(k+1)*batch_size]
            jp_texts = jp_corpus[k*batch_size:(k+1)*batch_size]

            for b in range(batch_size):
                en_size = len(en_texts[b])
                jp_size = len(jp_texts[b])

                if en_size > jp_size:
                    context_size = en_size
                else:
                    context_size = jp_size
                # print(context_size)

                inputs1 = np.zeros((context_size, en_vocab_size))
                inputs2 = np.zeros((context_size, jp_vocab_size))
                targets = np.zeros((context_size, jp_vocab_size))

                for i in range(en_size):
                    inputs1[i, en2id[en_texts[b][i]]] = 1

                for i in range(en_size, context_size):
                    inputs1[i, en2id[PAD]] = 1

                for i in range(jp_size):
                    inputs2[i, jp2id[jp_texts[b][i]]] = 1

                for i in range(jp_size, context_size):
                    inputs2[i, jp2id[PAD]] = 1

                for i in range(jp_size-1):
                    targets[i, jp2id[jp_texts[b][i+1]]] = 1

                tf.forward1(inputs1)
                outputs = tf.forward2(inputs2)
                delta = -targets / outputs
                loss[t] += np.sum(-targets * np.log(outputs))
                tf.backward(delta)
                tf.gradient()

            tf.adam(lr)
            tf.zero_gradient()
        end = time.time()
        print(f"経過時間: {end - start} 秒")
        #489.8336157798767 秒
        print(loss[t])

    tf.save()
    with open("loss.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(t + 1):
            writer.writerow([loss[i]])
    token_size = len(en_corpus[0])
    inputs = np.zeros((token_size, en_vocab_size))
    for i in range(token_size):
        inputs[i, en2id[en_corpus[0][i]]] = 1

    japanese = np.zeros((token_size, jp_vocab_size))
    japanese[0, jp2id[BOS]] = 1
    tf.forward1(inputs)

    for i in range(token_size-1):
        out = tf.forward2(japanese)
        select = np.argmax(out, axis=1)
        japanese[i+1, select[i]] = 1
        print(id2jp[select[i]])
