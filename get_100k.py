import pickle

import torch
from tqdm import tqdm
import esm
import random
import pandas as pd

noisy_prot_pep_df_path = 'Noisy_Dataset.csv'
noisy_df = pd.read_csv(noisy_prot_pep_df_path)
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
model.cuda() #push model to gpu

def generate_peptides(min_length = 15, max_length = 18, n = 5000,\
                      num_base_peps = 100, df = noisy_df, sample_variances = range(5,22, 4)):
  # 从数据集中筛选合适长度的肽序列
  base_peptides = df.loc[(df['pep_len'] <= max_length) & (df['pep_len'] >= min_length)].pep_seq.to_list()
  # 随机抽取 num_base_peps 个肽作为基础序列
  sampled_peptides = random.sample(base_peptides, num_base_peps)
  generated_peptides = []
  # 遍历基础肽，计算 ESM 嵌入
  for pep in tqdm(sampled_peptides):
    ##GET ESM EMBEDDING
    target_seq = pep
    batch_labels, batch_strs, batch_tokens = batch_converter([("target_seq", target_seq)])
    batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
      results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33].cpu()
    del batch_tokens
    # 计算每个基础肽需要生成的变体数量
    num_samples_per_base = int(int(n / num_base_peps) / 5)
    # 遍历不同高斯噪声
    for i in sample_variances:
      # 遍历每个高斯的生成数
      for j in range(num_samples_per_base):
        gen_pep = token_representations + torch.randn(token_representations.shape) * i * token_representations.var()
        # 给出氨基酸和对应所i你
        aa_toks = list("ARNDCEQGHILKMFPSTWYV")
        aa_idxs = [alphabet.get_idx(aa) for aa in aa_toks]
        #获取氨基酸中概率
        aa_logits = model.lm_head(gen_pep.cuda())[:, :, aa_idxs]
        # 转化出最有可能的氨基酸列表
        predictions = torch.argmax(aa_logits, dim=2).tolist()[0]
        generated_pep_seq = "".join([aa_toks[i] for i in predictions])
        generated_peptides.append(generated_pep_seq[1:-1])
  return generated_peptides

def generate_emb(seqs):
  for seq in seqs:
    batch_labels, batch_strs, batch_tokens = batch_converter([("seq", seq)])
    batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
      results = model(batch_tokens, repr_layers=[33], return_contacts=False)

##peptide search space size
num_base_peps = 1000
num_peps_per_base = 100
de_novo_peptides = generate_peptides(min_length = 15, max_length = 18, n = num_base_peps*num_peps_per_base, num_base_peps = num_base_peps)

candidate_peptide_dict = {}

for candidate_peptide in tqdm(de_novo_peptides):
  # 转化为token
  batch_labels, batch_strs, batch_tokens = batch_converter([("candidate_peptide", candidate_peptide)])
  batch_tokens = batch_tokens.cuda()
  batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

  # 获取嵌入结果
  with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)

  # 获取最后的表征
  token_representations = results["representations"][33].cpu()
  del batch_tokens

  sequence_representations = []
  for j, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[j, 1 : tokens_len - 1].mean(0))
  candidate_peptide_embedding = sequence_representations[0]
  candidate_peptide_dict.update({candidate_peptide:candidate_peptide_embedding})
output_file = "canonical_100k_denovo_peptides.pkl"

with open(output_file, "wb") as f:
  pickle.dump(candidate_peptide_dict, f)

all_candidate_peptides = list(candidate_peptide_dict.keys())