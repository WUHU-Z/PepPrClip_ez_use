import pandas as pd
from tqdm import tqdm
import pickle
import torch
import esm
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import date

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
model.cuda() #push model to gpu
noisy_prot_pep_df_path = 'Noisy_Dataset.csv'
strict_prot_pep_df_path = 'Strict_Dataset.csv'
noisy_df = pd.read_csv(noisy_prot_pep_df_path)
strict_df = pd.read_csv(strict_prot_pep_df_path)

def generate_peptides(min_length = 15, max_length = 18, n = 5000,num_base_peps = 100, df = noisy_df, sample_variances = range(5,22, 4)):
  # 从 df 数据集中筛选出特定长度（15-18）的肽序列
  base_peptides = df.loc[(df['pep_len'] <= max_length) & (df['pep_len'] >= min_length)].pep_seq.to_list()

  ##randomly sample 100 of these to use as base points
  sampled_peptides = random.sample(base_peptides, num_base_peps)

  ##embed with ESM, add random Gaussian noise on the order of
  #5, 9, 13, 17, and 21 standard deviations

  generated_peptides = []
  for pep in tqdm(sampled_peptides):
    ##GET ESM EMBEDDING
    target_seq = pep

    batch_labels, batch_strs, batch_tokens = batch_converter([("target_seq", target_seq)])
    batch_tokens = batch_tokens.cuda()
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
      results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33].cpu()
    del batch_tokens

    num_samples_per_base = int(int(n / num_base_peps) / 5)

    for i in sample_variances:
      for j in range(num_samples_per_base):
        gen_pep = token_representations + torch.randn(token_representations.shape) * i * token_representations.var()
        aa_toks = list("ARNDCEQGHILKMFPSTWYV")
        aa_idxs = [alphabet.get_idx(aa) for aa in aa_toks]
        aa_logits = model.lm_head(gen_pep.cuda())[:, :, aa_idxs]
        predictions = torch.argmax(aa_logits, dim=2).tolist()[0]
        generated_pep_seq = "".join([aa_toks[i] for i in predictions])
        generated_peptides.append(generated_pep_seq[1:-1])

  return generated_peptides

peptide_embeddings_path = "Pep_ESM_Embeddings.pickle"
protein_embeddings_path = "Prot_ESM_Embeddings.pickle"

with open(peptide_embeddings_path, 'rb') as file:
  pep_dict = pickle.load(file)

with open(protein_embeddings_path, 'rb') as file:
  prot_dict = pickle.load(file)

class PepProtDataset(torch.utils.data.Dataset):
  def __init__(self, pep_dict, prot_dict, pep_prot_df):
    super().__init__()
    self.pep_dict = pep_dict #存储肽（peptide）的嵌入
    self.prot_dict = prot_dict #存储蛋白质（protein）的嵌入
    self.pep_prot_df = pep_prot_df #存储配对信息

  def __len__(self):
      return len(self.pep_prot_df)  ##肽-蛋白数据对的数量

  # 获取单个样本
  def __getitem__(self, index):
    pdb_id, pep_seq, prot_seq, original_index = self.pep_prot_df.loc[index][['entry_id', 'pep_seq', 'prot_seq', 'index']]
    pep_esm_embedding = pep_dict[original_index]
    prot_esm_embedding = prot_dict[original_index]

    return_dict = {
        "index":original_index,
        "pdb_id": pdb_id,
        "pep_seq": pep_seq,
        "prot_seq": prot_seq,
        "peptide_input": pep_esm_embedding,
        "protein_input": prot_esm_embedding,
    }

    return return_dict

class PepProtDataModule(pl.LightningDataModule):
  def __init__(self, noisy_df, strict_df, batch_size=64):
    super().__init__()
    self.batch_size = batch_size #包含噪声数据的 DataFrame，训练时主要使用该数据
    self.noisy_df = noisy_df #严格筛选的数据
    self.strict_df = strict_df #批量大小

  def prepare_data(self):
    # train_df：从 noisy_df 选择前 2300 个 prot_cluster 作为训练集。
    # val_df：选取 2300-2600 的 prot_cluster 作为验证集。
    # test_df：选取 2600-3300 的 prot_cluster 作为测试集
    self.train_df = self.noisy_df.loc[
      self.noisy_df['prot_cluster'].isin(list(self.noisy_df.drop_duplicates('prot_cluster').prot_cluster)[0:2300])]
    self.val_df = self.noisy_df.loc[
      self.noisy_df['prot_cluster'].isin(list(self.noisy_df.drop_duplicates('prot_cluster').prot_cluster)[2300:2600])]
    self.test_df = self.noisy_df.loc[
      self.noisy_df['prot_cluster'].isin(list(self.noisy_df.drop_duplicates('prot_cluster').prot_cluster)[2600:3300])]
    # strict_val_df：严格数据集的验证集（2300-2600）。
    # strict_test_df：严格数据集的测试集（2600-3300）。
    self.strict_val_df = self.strict_df.loc[self.strict_df['prot_cluster'].isin(
      list(self.noisy_df.drop_duplicates('prot_cluster').prot_cluster)[2300:2600])]
    self.strict_test_df = self.strict_df.loc[self.strict_df['prot_cluster'].isin(
      list(self.noisy_df.drop_duplicates('prot_cluster').prot_cluster)[2600:3300])]

  #  初始化数据集（Dataset
  def setup(self, stage):
    self.train_dataset = PepProtDataset(pep_dict, prot_dict, self.train_df.reset_index())
    self.val_dataset = PepProtDataset(pep_dict, prot_dict, self.val_df.reset_index())
    self.test_dataset = PepProtDataset(pep_dict, prot_dict, self.test_df.reset_index())
    self.strict_val_dataset = PepProtDataset(pep_dict, prot_dict, self.strict_val_df.reset_index())
    self.strict_test_dataset = PepProtDataset(pep_dict, prot_dict, self.strict_test_df.reset_index())

  # 训练数据加载器
  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

  # 验证数据加载器
  def val_dataloader(self):
    full_batch = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
    binary_batch = DataLoader(self.val_dataset, batch_size=2, shuffle=True, drop_last=True)
    strict_full_batch = DataLoader(self.strict_val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
    strict_binary_batch = DataLoader(self.strict_val_dataset, batch_size=2, shuffle=True, drop_last=True)
    return [full_batch, binary_batch, strict_full_batch, strict_binary_batch]

  # 测试数据加载器
  def test_dataloader(self):
    full_batch = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
    binary_batch = DataLoader(self.test_dataset, batch_size=2, shuffle=True, drop_last=True)
    strict_full_batch = DataLoader(self.strict_test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
    strict_binary_batch = DataLoader(self.strict_test_dataset, batch_size=2, shuffle=True, drop_last=True)
    return [full_batch, binary_batch, strict_full_batch, strict_binary_batch]

class MiniCLIP(pl.LightningModule):
  def __init__(self, lr):
    super().__init__()
    self.lr = lr
    ##protein encoding: 2 layers, latent space of size 320?

    # 蛋白质嵌入网络
    self.prot_embedder = nn.Sequential(
      nn.Linear(1280, 640),
      nn.ReLU(),
      nn.Linear(640, 320),
    )

    # 肽嵌入网络
    ##peptide encoding: start with 2 layers, may want to add in a decoder later
    self.pep_embedder = nn.Sequential(
      nn.Linear(1280, 640),
      nn.ReLU(),
      nn.Linear(640, 320),
    )
  # 前向传播 (forward 方法)
  def forward(self, pep_input, prot_input):
    ##get peptide and protein embeddings, dot together
    pep_embedding = F.normalize(self.pep_embedder(pep_input))
    prot_embedding = F.normalize(self.prot_embedder(prot_input))

    logits = torch.matmul(pep_embedding, prot_embedding.T)  ##may need to transpose something here

    return logits

  def training_step(self, batch, batch_idx):

    logits = self(
      batch['peptide_input'],
      batch['protein_input'],
    )

    batch_size = batch['peptide_input'].shape[0]
    # 第 i 个 peptide 和第 j 个 protein 的匹配得分,所以labels[i] = i
    labels = torch.arange(batch_size).to(self.device)  ##NOTE: to(self.device) is important here
    ##this gives us the diagonal clip loss structure

    # loss of predicting partner using peptide
    partner_prediction_loss = F.cross_entropy(logits, labels)

    # loss of predicting peptide using partner
    peptide_prediction_loss = F.cross_entropy(logits.T, labels)

    loss = (partner_prediction_loss + peptide_prediction_loss) / 2

    self.log("train_loss", loss, sync_dist=True, batch_size=logits.shape[0])
    self.log("train_partner_prediction_loss", partner_prediction_loss, sync_dist=True, prog_bar=False,
             batch_size=logits.shape[0])
    self.log("train_peptide_prediction_loss", peptide_prediction_loss, sync_dist=True, prog_bar=False,
             batch_size=logits.shape[0])

    return loss

  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    # 对整个 batch 进行验证
    if dataloader_idx == 0 or dataloader_idx == 2:
      if dataloader_idx == 0:
        prefix = "noisy"
      else:
        prefix = "strict"

      # Predict on random batches of training batch size
      # 前向传播
      logits = self(
        batch['peptide_input'],
        batch['protein_input'],
      )

      batch_size = batch['peptide_input'].shape[0]
      labels = torch.arange(batch_size).to(self.device)  ##NOTE: to(self.device) is important here
      ##this gives us the diagonal clip loss structure

      # loss of predicting partner using peptide
      partner_prediction_loss = F.cross_entropy(logits, labels)

      # loss of predicting peptide using partner
      peptide_prediction_loss = F.cross_entropy(logits.T, labels)

      loss = (partner_prediction_loss + peptide_prediction_loss) / 2

      #  计算预测结果
      # # 预测 protein 的 peptide
      peptide_predictions = logits.argmax(dim=0)
      # 预测 peptide 的 protein
      partner_predictions = logits.argmax(dim=1)

      peptide_ranks = logits.argsort(dim=0).diag() + 1
      peptide_mrr = (peptide_ranks).float().pow(-1).mean()

      partner_ranks = logits.argsort(dim=1).diag() + 1
      partner_mrr = (partner_ranks).float().pow(-1).mean()

      partner_accuracy = partner_predictions.eq(labels).float().mean()
      peptide_accuracy = peptide_predictions.eq(labels).float().mean()

      k = int(logits.shape[0] / 10)
      peptide_topk_accuracy = torch.any((logits.topk(k, dim=0).indices - labels.reshape(1, -1)) == 0, dim=0).sum() / \
                              logits.shape[0]
      partner_topk_accuracy = torch.any((logits.topk(k, dim=1).indices - labels.reshape(-1, 1)) == 0, dim=1).sum() / \
                              logits.shape[0]
      # 计算top_K准确率并记录
      self.log(f"{prefix}_val_loss", loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0],
               add_dataloader_idx=False)
      self.log(f"{prefix}_val_perplexity", torch.exp(loss), sync_dist=False, prog_bar=True,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_val_partner_prediction_loss", partner_prediction_loss, sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_val_peptide_prediction_loss", peptide_prediction_loss, sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_val_partner_perplexity", torch.exp(partner_prediction_loss), sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_val_peptide_perplexity", torch.exp(peptide_prediction_loss), sync_dist=True, prog_bar=True,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_val_partner_accuracy", partner_accuracy, sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_val_peptide_accuracy", peptide_accuracy, sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_val_partner_top10p", partner_topk_accuracy, sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_val_peptide_top10p", peptide_topk_accuracy, sync_dist=True, prog_bar=True,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_val_peptide_mrr", peptide_mrr, sync_dist=True, prog_bar=False, batch_size=logits.shape[0],
               add_dataloader_idx=False)
      self.log(f"{prefix}_val_partner_mrr", partner_mrr, sync_dist=True, prog_bar=False, batch_size=logits.shape[0],
               add_dataloader_idx=False)

    # 二分类任务
    else:
      if dataloader_idx == 1:
        prefix = "noisy"
      else:
        prefix = "strict"

      # Given a protein, predict the correct peptide out of 2
      logits = self(
        batch['peptide_input'],
        batch['protein_input'],
      )

      batch_size = batch['peptide_input'].shape[0]
      labels = torch.arange(batch_size).to(self.device)  ##NOTE: to(self.device) is important here
      ##this gives us the diagonal clip loss structure

      binary_cross_entropy = F.cross_entropy(logits.T, labels)

      binary_predictions = logits.argmax(dim=0)
      binary_accuracy = binary_predictions.eq(labels).float().mean()

      self.log(f"{prefix}_binary_loss", binary_cross_entropy, sync_dist=True, prog_bar=False, batch_size=2,
               add_dataloader_idx=False)
      self.log(f"{prefix}_binary_accuracy", binary_accuracy, sync_dist=False, prog_bar=True, batch_size=2,
               add_dataloader_idx=False)

  # 处理测试集
  def test_step(self, batch, batch_idx, dataloader_idx=0):

    if dataloader_idx == 0 or dataloader_idx == 2:
      if dataloader_idx == 0:
        prefix = "noisy"
      else:
        prefix = "strict"

      # Predict on random batches of training batch size
      logits = self(
        batch['peptide_input'],
        batch['protein_input'],
      )

      batch_size = batch['peptide_input'].shape[0]
      labels = torch.arange(batch_size).to(self.device)  ##NOTE: to(self.device) is important here
      ##this gives us the diagonal clip loss structure

      # loss of predicting partner using peptide
      partner_prediction_loss = F.cross_entropy(logits, labels)

      # loss of predicting peptide using partner
      peptide_prediction_loss = F.cross_entropy(logits.T, labels)

      loss = (partner_prediction_loss + peptide_prediction_loss) / 2

      # prediction of peptides for each partner
      peptide_predictions = logits.argmax(dim=0)
      # prediction of partners for each peptide
      partner_predictions = logits.argmax(dim=1)

      peptide_ranks = logits.argsort(dim=0).diag() + 1
      peptide_mrr = (peptide_ranks).float().pow(-1).mean()

      partner_ranks = logits.argsort(dim=1).diag() + 1
      partner_mrr = (partner_ranks).float().pow(-1).mean()

      partner_accuracy = partner_predictions.eq(labels).float().mean()
      peptide_accuracy = peptide_predictions.eq(labels).float().mean()

      k = int(logits.shape[0] / 10)
      peptide_topk_accuracy = torch.any((logits.topk(k, dim=0).indices - labels.reshape(1, -1)) == 0, dim=0).sum() / \
                              logits.shape[0]
      partner_topk_accuracy = torch.any((logits.topk(k, dim=1).indices - labels.reshape(-1, 1)) == 0, dim=1).sum() / \
                              logits.shape[0]

      self.log(f"{prefix}_test_loss", loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0],
               add_dataloader_idx=False)
      self.log(f"{prefix}_test_perplexity", torch.exp(loss), sync_dist=False, prog_bar=True,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_test_partner_prediction_loss", partner_prediction_loss, sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_test_peptide_prediction_loss", peptide_prediction_loss, sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_test_partner_perplexity", torch.exp(partner_prediction_loss), sync_dist=True,
               prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_test_peptide_perplexity", torch.exp(peptide_prediction_loss), sync_dist=True, prog_bar=True,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_test_partner_accuracy", partner_accuracy, sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_test_peptide_accuracy", peptide_accuracy, sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_test_partner_top10p", partner_topk_accuracy, sync_dist=True, prog_bar=False,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_test_peptide_top10p", peptide_topk_accuracy, sync_dist=True, prog_bar=True,
               batch_size=logits.shape[0], add_dataloader_idx=False)
      self.log(f"{prefix}_test_peptide_mrr", peptide_mrr, sync_dist=True, prog_bar=False, batch_size=logits.shape[0],
               add_dataloader_idx=False)
      self.log(f"{prefix}_test_partner_mrr", partner_mrr, sync_dist=True, prog_bar=False, batch_size=logits.shape[0],
               add_dataloader_idx=False)

    # 二分类
    else:
      if dataloader_idx == 1:
        prefix = "noisy"
      else:
        prefix = "strict"

      # Given a protein, predict the correct peptide out of 2
      logits = self(
        batch['peptide_input'],
        batch['protein_input'],
      )

      batch_size = batch['peptide_input'].shape[0]
      labels = torch.arange(batch_size).to(self.device)  ##NOTE: to(self.device) is important here
      ##this gives us the diagonal clip loss structure

      binary_cross_entropy = F.cross_entropy(logits.T, labels)

      binary_predictions = logits.argmax(dim=0)
      binary_accuracy = binary_predictions.eq(labels).float().mean()

      self.log(f"{prefix}_test_binary_loss", binary_cross_entropy, sync_dist=True, prog_bar=False, batch_size=2,
               add_dataloader_idx=False)
      self.log(f"{prefix}_test_binary_accuracy", binary_accuracy, sync_dist=False, prog_bar=True, batch_size=2,
               add_dataloader_idx=False)

  # 自定义优化器
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# 定义 Early Stopping 早停机制
early_stop_callback = EarlyStopping(
   monitor='strict_val_loss',
   min_delta=0.00,
   patience=3,
   verbose=False,
   mode='min'
)
print('加载数据集')
datamodule = PepProtDataModule(noisy_df, strict_df)
miniclip = MiniCLIP(lr = 0.003)
trainer = pl.Trainer(callbacks=[early_stop_callback])
print('开始训练')
trainer.fit(miniclip, datamodule=datamodule)
print('运行验证集')
trainer.validate(miniclip, datamodule=datamodule)
print('运行测试集')
trainer.test(miniclip, datamodule=datamodule)
today = date.today()
print('权重保存')
trainer.save_checkpoint(f"pepprclip_{today}.ckpt")
