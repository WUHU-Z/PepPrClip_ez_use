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
import pandas as pd

noisy_prot_pep_df_path = 'Noisy_Dataset.csv'
noisy_df = pd.read_csv(noisy_prot_pep_df_path)

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
model.cuda() #push model to gpu

def generate_peptides(min_length = 15, max_length = 18, n = 5000,\
                      num_base_peps = 100, df = noisy_df, sample_variances = range(5,22, 4)):

  base_peptides = df.loc[(df['pep_len'] <= max_length) & (df['pep_len'] >= min_length)].pep_seq.to_list()
  sampled_peptides = random.sample(base_peptides, num_base_peps)
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

    for i in tqdm(sample_variances):
      for j in range(num_samples_per_base):
        gen_pep = token_representations + torch.randn(token_representations.shape) * i * token_representations.var()
        aa_toks = list("ARNDCEQGHILKMFPSTWYV")
        aa_idxs = [alphabet.get_idx(aa) for aa in aa_toks]
        aa_logits = model.lm_head(gen_pep.cuda())[:, :, aa_idxs]
        predictions = torch.argmax(aa_logits, dim=2).tolist()[0]
        generated_pep_seq = "".join([aa_toks[i] for i in predictions])
        generated_peptides.append(generated_pep_seq[1:-1])

  return generated_peptides

class MiniCLIP(pl.LightningModule):
  def __init__(self, lr):
    super().__init__()
    self.lr = lr
    ##protein encoding: 2 layers, latent space of size 320?

    self.prot_embedder = nn.Sequential(
      nn.Linear(1280, 640),
      nn.ReLU(),
      nn.Linear(640, 320),
    )

    ##peptide encoding: start with 2 layers, may want to add in a decoder later
    self.pep_embedder = nn.Sequential(
      nn.Linear(1280, 640),
      nn.ReLU(),
      nn.Linear(640, 320),
    )

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

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)

miniclip = MiniCLIP.load_from_checkpoint("pepprclip_2025-03-09.ckpt", lr = 0.003)
miniclip.eval()

peps_per_target = 8
EWSFLI1 = 'MASTDYSTYSQAAAQQGYSAYTAQPTQGYAQTTQAYGQQSYGTYGQPTDVSYTQAQTTATYGQTAYATSYGQPPTGYTTPTAPQAYSQPVQGYGTGAYDTTTATVTTTQASYAAQSAYGTQPAYPAYGQQPAATAPTRPQDGNKPTETSQPQSSTGGYNQPSLGYGQSNYSYPQVPGSYPMQPVTAPPSYPPTSYSSTQPTSYDQSSYSQQNTYGQPSSYGQQSSYGQQSSYGQQPPTSYPPQTGSYSQAPSQYSQQSSSYGQQNPSYDSVRRGAWGNNMNSGLNKSPPLGGAQTISKNTEQRPQPDPYQILGPTSSRLANPGSGQIQLWQFLLELLSDSANASCITWEGTNGEFKMTDPDEVARRWGERKSKPNMNYDKLSRALRYYYDKNIMTKVHGKRYAYKFDFHGIAQALQPHPTESSMYKYPSDISYMPSYHAHQQKVNFVPPHPSSMPVTSSSFFGAASQYWTSPTGGIYPNPNVPRHPNTHVPSHLGSYY'
AMHR2 = 'MLGSLGLWALLPTAVEAPPNRRTCVFFEAPGVRGSTKTLGELLDTGTELPRAIRCLYSRCCFGIWNLTQDRAQVEMQGCRDSDEPGCESLHCDPSPRAHPSPGSTLFTCSCGTDFCNANYSHLPPPGSPGTPGSQGPQAAPGESIWMALVLLGLFLLLLLLLGSIILALLQRKNYRVRGEPVPEPRPDSGRDWSVELQELPELCFSQVIREGGHAVVWAGQLQGKLVAIKAFPPRSVAQFQAERALYELPGLQHDHIVRFITASRGGPGRLLSGPLLVLELHPKGSLCHYLTQYTSDWGSSLRMALSLAQGLAFLHEERWQNGQYKPGIAHRDLSSQNVLIREDGSCAIGDLGLALVLPGLTQPPAWTPTQPQGPAAIMEAGTQRYMAPELLDKTLDLQDWGMALRRADIYSLALLLWEILSRCPDLRPDSSPPPFQLAYEAELGNTPTSDELWALAVQERRRPYIPSTWRCFATDPDGLRELLEDCWDADPEARLTAECVQQRLAALAHPQESHPFPESCPRGCPPLCPEDCTSIPAPTILPCRPQRSACHFSVQQGPCSRNPQPACTLSPV'
FOXP3 ='MPNPRPGKPSAPSLALGPSPGASPSWRAAPKASDLLGARGPGGTFQGRDLRGGAHASSSSLNPMPPSQLQLPTLPLVMVAPSGARLGPLPHLQALLQDRPHFMHQLSTVDAHARTPVLQVHPLESPAMISLTPPTTATGVFSLKARPGLPPGINVASLEWVSREPALLCTFPNPSAPRKDSTLSAVPQSSYPLLANGVCKWPGCEKVFEEPEDFLKHCQADHLLDEKGRAQCLLQREMVQSLEQQLVLEKEKLSAMQAHLAGKMALTKASSVASSDKGSCCIVAAGSQGPVVPAWSGPREAPDSLFAVRRHLWGSHGNSTFPEFLHNMDYFKFHNMRPPFTYATLIRWAILEAPEKQRTLNEIYHWFTRMFAFFRNHPATWKNAIRHNLSLHKCFVRVESEKGAVWTVDELEFRKKRSQRPSRCSNPTPGP'
TRIM8 ='MAENWKNCFEEELICPICLHVFVEPVQLPCKHNFCRGCIGEAWAKDSGLVRCPECNQAYNQKPGLEKNLKLTNIVEKFNALHVEKPPAALHCVFCRRGPPLPAQKVCLRCEAPCCQSHVQTHLQQPSTARGHLLVEADDVRAWSCPQHNAYRLYHCEAEQVAVCQYCCYYSGAHQGHSVCDVEIRRNEIRKMLMKQQDRLEEREQDIEDQLYKLESDKRLVEEKVNQLKEEVRLQYEKLHQLLDEDLRQTVEVLDKAQAKFCSENAAQALHLGERMQEAKKLLGSLQLLFDKTEDVSFMKNTKSVKILMDRTQTCTSSSLSPTKIGHLNSKLFLNEVAKKEKQLRKMLEGPFSTPVPFLQSVPLYPCGVSSSGAEKRKHSTAFPEASFLETSSGPVGGQYGAAGTASGEGQSGQPLGPCSSTQHLVALPGGAQPVHSSPVFPPSQYPNGSAAQQPMLPQYGGRKILVCSVDNCYCSSVANHGGHQPYPRSGHFPWTVPSQEYSHPLPPTPSVPQSLPSLAVRDWLDASQQPGHQDFYRVYGQPSTKHYVTS'
Betacatenin = 'MATQADLMELDMAMEPDRKAAVSHWQQQSYLDSGIHSGATTTAPSLSGKGNPEEEDVDTSQVLYEWEQGFSQSFTQEQVADIDGQYAMTRAQRVRAAMFPETLDEGMQIPSTQFDAAHPTNVQRLAEPSQMLKHAVVNLINYQDDAELATRAIPELTKLLNDEDQVVVNKAAVMVHQLSKKEASRHAIMRSPQMVSAIVRTMQNTNDVETARCTAGTLHNLSHHREGLLAIFKSGGIPALVKMLGSPVDSVLFYAITTLHNLLLHQEGAKMAVRLAGGLQKMVALLNKTNVKFLAITTDCLQILAYGNQESKLIILASGGPQALVNIMRTYTYEKLLWTTSRVLKVLSVCSSNKPAIVEAGGMQALGLHLTDPSQRLVQNCLWTLRNLSDAATKQEGMEGLLGTLVQLLGSDDINVVTCAAGILSNLTCNNYKNKMMVCQVGGIEALVRTVLRAGDREDITEPAICALRHLTSRHQEAEMAQNAVRLHYGLPVVVKLLHPPSHWPLIKATVGLIRNLALCPANHAPLREQGAIPRLVQLLVRAHQDTQRRTSMGGTQQQFVEGVRMEEIVEGCTGALHILARDVHNRIVIRGLNTIPLFVQLLYSPIENIQRVAAGVLCELAQDKEAAEAIEAEGATAPLTELLHSRNEGVATYAAAVLFRMSEDKPQDYKKRLSVELTSSLFRTEPMAWNETADLGLDIGAQGEPLGYRQDDPSYRSFHSGGYGQDALGMDPMMEHEMGGHHPGADYPVDGLPDLGHAQDLMDGLPPGDSNQLAWFDTDL'
ETV6 = 'MSETPAQCSIKQERISYTPPESPVPSYASSTPLHVPVPRALRMEEDSIRLPAHLRLQPIYWSRDDVAQWLKWAENEFSLRPIDSNTFEMNGKALLLLTKEDFRYRSPHSGDVLYELLQHILKQRKPRILFSPFFHPGNSIHTQPEVILHQNHEEDNCVQRTPRPSVDNVHHNPPTIELLHRSRSPITTNHRPSPDPEQRPLRSPLDNMIRRLSPAERAQGPRPHQENNHQESYPLSVSPMENNHCPASSESHPKPSSPRQESTRVIQLMPSPIMHPLILNPRHSVDFKQSRLSEDGLHREGKPINLSHREDLAYMNHIMVSVSPPEEHAMPIGRIADCRLLWDYVYQLLSDSRYENFIRWEDKESKIFRIVDPNGLARLWGNHKNRTNMTYEKMSRALRHYYKLNIIRKEPGQRLLFRFMKTPDEIMSGRTDRLEHLESQELDEQIYQEDEC'
PAX3FOXO1 = 'MTTLAGAVPRMMRPGPGQNYPRSGFPLEVSTPLGQGRVNQLGGVFINGRPLPNHIRHKIVEMAHHGIRPCVISRQLRVSHGCVSKILCRYQETGSIRPGAIGGSKPKQVTTPDVEKKIEEYKRENPGMFSWEIRDKLLKDAVCDRNTVPSVSSISRILRSKFGKGEEEEADLERKEAEESEKKAKHSIDGILSERASAPQSDEGSDIDSEPDLPLKRKQRRSRTTFTAEQLEELERAFERTHYPDIYTREELAQRAKLTEARVQVWFSNRRARWRKQAGANQLMAFNHLIPGGFPPTAMPTLPTYQLSETSYQPTSIPQAVSDPSSTVHRPQPLPPSTVHQSTIPSNPDSSSAYCLPSTRHGFSSYTDSFVPPSGPSNPMNPTIGNGLSPQNSIRHNLSLHSKFIRVQNEGTGKSSWWMLNPEGGKSGKSPRRRAASMDNNSKFAKSRSRAAKKKASLQSGQEGAGDSPGSQFSKWPASPGSHSNDDFDNWSTFRPRTSSNASTISGRLSPIMTEQDDLGEGDVHSMVYPPSAAKMASTLPSLSEISNPENMENLLDNLNLLSSPTSLTVSTQSSPGTMMQQTPCYSFAPPNTSLNSPSPNYQKYTYGQSSMSPLPQMPIQTLQDNKSSYGGMSQYNCAPGLLKELLTSDSPPHNDIMTPVDPGVAQPNSRVLGQNVMMGPNSVMSTYGSQASHNKMMNPSSHTHPGHAQQTSAVNGRPLPHTVSTMPHTSGMNRLTQVKTPVQVPLPHPMQMSALGGYSSVSSCNGYGRMGLLHQEKLPSDLDGMFIERLDCDMESIIRNDLMDGDTLDFNFDNVLPNQSFPHSVKTTTHSWVSG'
targets = {'EWSFLI1':EWSFLI1,'AMHR2':AMHR2, 'FOXP3':FOXP3,'TRIM8':TRIM8,'Betacatenin':Betacatenin, 'ETV6':ETV6, 'PAX3FOXO1':PAX3FOXO1}

# 存储所有候选肽的序列
with open("canonical_100k_denovo_peptides.pkl", "rb") as f:
  candidate_peptide_dict = pickle.load(f)
all_candidate_peptides = list(candidate_peptide_dict.keys())

output_dict = {}

for name, target_seq in tqdm(targets.items()):
  ##feed sequence it into ESM
  batch_labels, batch_strs, batch_tokens = batch_converter([("target_seq", target_seq)])
  batch_tokens = batch_tokens.cuda()
  batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

  with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)

  token_representations = results["representations"][33].cpu()
  del batch_tokens

  sequence_representations = []
  # 取出 去除 <CLS> 和 <EOS> 标记后的所有 token 向量的均值，得到目标蛋白的最终向量表示
  for j, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[j, 1 : tokens_len - 1].mean(0))

  target_prot_embedding = sequence_representations[0]
  print('靶标的大小',target_prot_embedding.shape)
  peptide_scores = []
  seq_to_score_dict = {}

  for candidate_peptide, candidate_peptide_embedding in tqdm(
    candidate_peptide_dict.items(),
    desc="Processing Peptides",
    mininterval=10  # 每 10 秒更新一次
  ):
      score = miniclip.forward(candidate_peptide_embedding.unsqueeze(0), target_prot_embedding.unsqueeze(0))
      peptide_scores.append(score)
      seq_to_score_dict.update({candidate_peptide:score})

  topk_idxs = list(torch.concat(peptide_scores).argsort(dim=0, descending=True)[:peps_per_target])
  topk_peptides = [all_candidate_peptides[topk_idxs[i]] for i in range(len(topk_idxs))]
  topk_scores = [float(seq_to_score_dict[peptide]) for peptide in topk_peptides]

  output_dict.update({name:(topk_peptides, topk_scores)})

