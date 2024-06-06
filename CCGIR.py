# coding:utf-8
import pickle
import faiss
import torch
import heapq
import numpy as np
import Levenshtein
import csv
from nlgeval import compute_metrics
from tqdm import tqdm

from transformers import RobertaTokenizer, RobertaModel
import pandas as pd

from bert_whitening import sents_to_vecs, transform_and_normalize
from similarities import BertSimilarity
mm = BertSimilarity(model_name_or_path="C:/pre_training_model/text2vec-base-multilingual")
# r = mm.similarity("It is good", "别的复苏步伐")
# r = mm.similarity(desc[108], desc[114])
# print(f"similarity score: {float(r)}")

dim = 256

df = pd.read_csv("./dataset/train/train_code.csv", header=None)
train_code_list = df[0].tolist()
df = pd.read_csv("./dataset/train/train_ast.csv", header=None)
train_ast_list = df[0].tolist()
df = pd.read_csv("./dataset/train/train_desc.csv", header=None)
train_nl_list = df[0].tolist()
df = pd.read_csv("./dataset/test/test_code.csv", header=None)
test_code_list = df[0].tolist()
df = pd.read_csv("./dataset/test/test_ast.csv", header=None)
test_ast_list = df[0].tolist()
df = pd.read_csv("./dataset/test/test_desc.csv", header=None, on_bad_lines='skip', sep = '\t',quoting=csv.QUOTE_NONE)
test_nl_list = df[0].tolist()

tokenizer = RobertaTokenizer.from_pretrained("C:/pre_training_model/codebert-base") #Local model file
model = RobertaModel.from_pretrained("C:/pre_training_model/codebert-base")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)

def sim_jaccard(s1, s2):
    """jaccard相似度"""
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

class Retrieval(object):
    def __init__(self):
        f = open('./model/code_vector_whitening.pkl', 'rb')
        self.bert_vec = pickle.load(f)
        f.close()
        f = open('./model/kernel.pkl', 'rb')
        self.kernel = pickle.load(f)
        f.close()
        f = open('./model/bias.pkl', 'rb')
        self.bias = pickle.load(f)
        f.close()

        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None

    def encode_file(self):
        all_texts = []
        all_ids = []
        all_vecs = []
        for i in range(len(train_code_list)):
            all_texts.append(train_code_list[i])
            all_ids.append(i)
            all_vecs.append(self.bert_vec[i].reshape(1,-1))
        all_vecs = np.concatenate(all_vecs, 0)
        id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
        self.id2text = id2text
        self.vecs = np.array(all_vecs, dtype="float32")
        self.ids = np.array(all_ids, dtype="int64")

    def build_index(self, n_list):
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, min(n_list, self.vecs.shape[0]))
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def single_query(self, code, ast, desc, topK):
        body = sents_to_vecs([code], tokenizer, model)
        body = transform_and_normalize(body, self.kernel, self.bias)
        vec = body[[0]].reshape(1, -1).astype('float32')
        _, sim_idx = self.index.search(vec, topK)
        sim_idx = sim_idx[0].tolist()
        sim_scores = []
        scores = []
        sim_nls = []
        # print(len(sim_idx))
        for j in sim_idx:
            if j >= len(train_ast_list):
                print("Index error: j=", j, "len(train_ast_list)=", len(train_ast_list))
                continue
            code_score = sim_jaccard(train_code_list[j].split(), code.split())
            ast_score = Levenshtein.seqratio(str(train_ast_list[j]).split(), str(ast).split())
            desc_score = float(mm.similarity(train_nl_list[j], desc))

            # sim_scores.append(0.7 * code_score + 0.3 * ast_score)

            scores.append(1.0*desc_score + 0.0*(0.7 * code_score + 0.3 * ast_score))

            sim_nls.append(train_nl_list[j])

  #      topk_idx = heapq.nlargest(topK, range(len(sim_scores)), key=sim_scores.__getitem__)
 #      topk_nls = [sim_nls[i] for i in topk_idx]
 #       topk_code= [ train_code_list[sim_idx[i]] for i in topk_idx]
    #    topk_score = [ sim_scores[i] for i in topk_idx]

        topk_idx2 = heapq.nlargest(topK, range(len(scores)), key=scores.__getitem__)
        topk_nls2 = [sim_nls[i] for i in topk_idx2]
        topk_code2 = [train_code_list[sim_idx[i]] for i in topk_idx2]
        topk_score2 = [scores[i] for i in topk_idx2]

        return topk_nls2,topk_code2,topk_score2

if __name__ == '__main__':
    ccgir = Retrieval()
    print("Sentences to vectors")
    ccgir.encode_file()
    print("加载索引")
    ccgir.build_index(n_list=1)
    ccgir.index.nprob = 1
    sim_nl_list, c_list, sim_score_list, nl_list ,sim_nl_codelist= [], [], [], [], []
    sim_nl_list2, c_list2, sim_score_list2, nl_list2, sim_nl_codelist2 = [], [], [], [], []
    data_list = []
    l = len(test_code_list)
    for i in tqdm(range(len(test_code_list))):
        sim_nls2, sim_code2 ,sim_score2  = ccgir.single_query(test_code_list[i], test_ast_list[i], test_nl_list[i],topK=10)

#        sim_nl_list.append(sim_nls)
 #       sim_nl_codelist.append(sim_code)
  #      nl_list.append(test_nl_list[i])
  #      sim_score_list.append(sim_score)


        sim_nl_list2.append(sim_nls2)
        sim_nl_codelist2.append(sim_code2)
        nl_list2.append(test_nl_list[i])
        sim_score_list2.append(sim_score2)

    # df = pd.DataFrame(nl_list)
    # df.to_csv("./results/9-1/nl.csv", index=False,header=None)
    # df = pd.DataFrame(sim_nl_codelist)
    # df.to_csv("./results/9-1/code-10.csv", index=False, header=None)
    # df = pd.DataFrame(sim_nl_list)
    # df.to_csv("./results/9-1/sim-10.csv", index=False,header=None)
    # df = pd.DataFrame(sim_score_list)
    # df.to_csv("./results/9-1/sim-10-score.csv", index=False, header=None)


    df = pd.DataFrame(nl_list2)
    df.to_csv("./results/10-0/nl_2.csv", index=False, header=None)
    df = pd.DataFrame(sim_nl_codelist2)
    df.to_csv("./results/10-0/code-10_2.csv", index=False, header=None)
    df = pd.DataFrame(sim_nl_list2)
    df.to_csv("./results/10-0/sim-10_2.csv", index=False, header=None)
    df = pd.DataFrame(sim_score_list2)
    df.to_csv("./results/10-0/sim-10-score_2.csv", index=False, header=None)
