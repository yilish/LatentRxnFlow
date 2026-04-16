from torch import nn
import torch
import torch.nn.functional as F
import math
import pdb
import os
from torch.nn import MultiheadAttention
from torch.distributions.multinomial import Multinomial

MAX_BONDS = 6
MAX_DIFF = 4

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings55 have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/dim))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/dim))
        \text{where pos is the word position and i is the embed idx)
    Args:
        dim: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """
    def __init__(self, dim, dropout=0.1, max_len = 512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe) # trainable

    def forward(self, l):
        r"""
        returns the additive embedding, notice that addition isnot done in this function
        input shape [l, b, ...] outputshape [l, 1, dim]
        """
        tmp = self.pe[:l, :]
        return self.dropout(tmp)

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, 4*dim, 1)
        self.conv2 = nn.Conv1d(4*dim, dim, 1)
        self.conv3 = nn.Conv1d(dim, 4*dim, 1)
        self.conv4 = nn.Conv1d(4*dim, dim, 1)
        self.conv5 = nn.Conv1d(dim, dim, 1)
        
    def forward(self, x):
        inter = self.conv1(x)
        inter = F.relu(inter)
        inter = self.conv2(inter)
        x = x + inter
        
        inter = self.conv3(x)
        inter = F.relu(inter)
        inter = self.conv4(inter)
        x = x + inter
        
        return self.conv5(x)

    
    
class AtomEncoder(nn.Module):
    def __init__(self, ntoken, dim, dropout=0.1, rank=0):
        super().__init__()
        self.position_embedding = PositionalEncoding(dim, dropout=dropout)
        self.element_embedding = nn.Embedding(ntoken, dim)
        self.charge_embedding = nn.Embedding(13, dim) #[-6, +6]
        self.aroma_embedding = nn.Embedding(2, dim)
        self.reactant_embedding = nn.Embedding(2, dim)
        self.segment_embedding = nn.Embedding(30, dim)
        self.rank = rank
        self.mlp = MLP(dim)
        
    def forward(self, element, bond, aroma, charge, segment, reactant_mask=None):
        '''
        element, long [b, l] element index
        bonds, long [b, l, MAX_BONDS]
        aroma, long [b, l]
        charge, long [b, l] +2 +1 0 -1 -2
        
        returns [l, b, dim]
        
        '''
        b, l = element.shape
        # basic information
        element = element.transpose(1, 0) 
        element_embedding = self.element_embedding(element)
        embedding = element_embedding
        #[l, b, dim]

        position_embedding = self.position_embedding(l)
        embedding = embedding + position_embedding
        
        aroma = aroma.transpose(1, 0).long()
        aroma_embedding = self.aroma_embedding(aroma)
        embedding = embedding + aroma_embedding
        
        # additional information
        charge = charge.transpose(1, 0) + 6  
        charge_embedding = self.charge_embedding(charge)
        embedding = embedding + charge_embedding
        
        segment = segment.transpose(1, 0) 
        segment_embedding = self.segment_embedding(segment)
        embedding = embedding + segment_embedding
        
        if not reactant_mask is None:
            reactant_mask = reactant_mask.transpose(1, 0) 
            reactant_embedding = self.reactant_embedding(reactant_mask)
            embedding = embedding + reactant_embedding  
            
        message = self.mlp(embedding.permute(1, 2, 0)).permute(2, 0, 1)
        device = bond.device
        eye = torch.eye(l, device=device, dtype=torch.float32)
        eye = eye.to(device)
        # eye = torch.eye(l).to(self.rank)
        tmp = torch.index_select(eye, dim=0, index=bond.reshape(-1)).view(b, l, MAX_BONDS, l).sum(dim=2) # adjacenct matrix
        tmp = tmp*(1-eye) # remove self loops
        message = torch.einsum("lbd,bkl->kbd", message, tmp)
        
        embedding = embedding + message
        
        return embedding


class BondDecoder(nn.Module):
    def __init__(self, dim, rank=0):
        super().__init__()
        self.inc_attention = MultiheadAttention(dim, MAX_DIFF)
        self.inc_q = nn.Conv1d(dim, dim, 1)
        self.inc_k = nn.Conv1d(dim, dim, 1)
        
        self.dec_attention = MultiheadAttention(dim, MAX_DIFF)
        self.dec_q = nn.Conv1d(dim, dim, 1)
        self.dec_k = nn.Conv1d(dim, dim, 1)

        self.rank = rank


    def forward(self, molecule_embedding, src_bond, src_mask, tgt_bond=None, tgt_mask=None, return_logits=False):
        """
            mask == True iff masked
            molecule_embedding of shape [l, b, dim]
        """
        l, b, dim = molecule_embedding.shape
        molecule_embedding = molecule_embedding.permute(1, 2, 0)  # to [b, dim, l]
        
        q, k, v = self.inc_q(molecule_embedding), self.inc_k(molecule_embedding), molecule_embedding
        q, k, v = q.permute(2, 0, 1), k.permute(2, 0, 1), v.permute(2, 0, 1)  # to [l, b, c]
        _, inc = self.inc_attention(q, k, v, key_padding_mask=src_mask)

        q, k, v = self.dec_q(molecule_embedding), self.dec_k(molecule_embedding), molecule_embedding
        q, k, v = q.permute(2, 0, 1), k.permute(2, 0, 1), v.permute(2, 0, 1)  # to [l, b, c]
        _, dec = self.dec_attention(q, k, v, key_padding_mask=src_mask)
        
        pad_mask = 1 - src_mask.float()
        # [B, L], 0 if padding
        pad_mask = torch.einsum("bl,bk->blk", pad_mask, pad_mask)
        diff = (inc - dec)*MAX_DIFF*pad_mask
        device = src_bond.device
        eye = torch.eye(src_mask.shape[1], device=device, dtype=torch.float32)
        eye = eye.to(device)
        # eye = torch.eye(src_mask.shape[1]).to(self.rank)
        src_weight = torch.index_select(eye, dim=0, index=src_bond.reshape(-1)).view(b, l, MAX_BONDS, l).sum(dim=2)* pad_mask
        pred_weight = src_weight + diff      
        if return_logits:
            pred_weight = (pred_weight + pred_weight.permute(0, 2, 1))/2
            return pred_weight
        if tgt_bond is None: # inference
            # [b, l, l]
            bonds = []
            pred_weight = (pred_weight + pred_weight.permute(0, 2, 1))/2
            for i in range(MAX_BONDS):
                bonds += [pred_weight.argmax(2)]
                pred_weight -= torch.index_select(eye, dim=0, index=bonds[-1].reshape(-1)).view(b, l, l)
            pred_bond = torch.stack(bonds, dim =2)
            return pred_bond
            
        else: # training
            tgt_mask = tgt_mask.float() # 1 iff masked
            or_mask = 1 - torch.einsum("bl,bk->blk", tgt_mask, tgt_mask) # notice that this doesn't mask the edges between target and side products
            and_mask = torch.einsum("bl,bk->blk", 1-tgt_mask, 1-tgt_mask)
        
            tgt_weight = torch.index_select(eye, dim=0, index=tgt_bond.reshape(-1)).view(b, l, MAX_BONDS, l).sum(dim=2)*and_mask
            error = pred_weight - tgt_weight
            error = error*error*pad_mask*or_mask
            loss = error.sum(dim=(1, 2))
            return {'bond_loss':loss}


from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import TransformerDecoder, TransformerDecoderLayer

from torch import nn
import torch
import torch.nn.functional as F
import math
import pdb
import os
from transformers import AutoModel

class MoleculeEncoder(nn.Module):
    def __init__(self, ntoken, dim, nhead, nlayer, dropout, rank, args):
        super().__init__()
        self.args = args
        self.atom_encoder = AtomEncoder(ntoken, dim, dropout=dropout, rank=rank)
        layer = TransformerEncoderLayer(dim, nhead, dim, dropout)
        self.transformer_encoder = TransformerEncoder(layer, nlayer)
        # multihead attention assumes [len, batch, dim]
        
        # padding_mask = True equivalent to mask = -inf

    def forward(self, element, bond, aroma, charge, mask, segment, reactant=None, action_embedding=None, solvent_embedding=None, catalyst_embedding=None):
        '''
        element, long [b, l] element index
        bonds, long [b, l, 4]
        aroma, long [b, l]
        charge, long [b, l] +1 0 -1
        mask, [b, l] true if masked
        returns [l, b, dim]
        '''
        embedding = self.atom_encoder(element, bond, aroma, charge, segment, reactant)
        if action_embedding is not None:
            embedding = torch.cat([embedding, action_embedding], dim=0)
            # print(mask.shape, action_embedding.shape)
            mask = torch.cat([mask, torch.zeros(mask.size(0), action_embedding.size(0)).bool().to(mask.device)], dim=1)
        if solvent_embedding is not None:
            embedding = torch.cat([embedding, solvent_embedding], dim=0)
            mask = torch.cat([mask, torch.zeros(mask.size(0), solvent_embedding.size(0)).bool().to(mask.device)], dim=1)
        if catalyst_embedding is not None:
            embedding = torch.cat([embedding, catalyst_embedding], dim=0)
            mask = torch.cat([mask, torch.zeros(mask.size(0), catalyst_embedding.size(0)).bool().to(mask.device)], dim=1)
        encoder_output = self.transformer_encoder(embedding, src_key_padding_mask=mask)
        return encoder_output


class VariationalEncoder(nn.Module):
    def __init__(self, dim, nhead, nlayer, dropout, rank=0):
        super().__init__()
        layer = TransformerDecoderLayer(dim, nhead, dim, dropout)
        self.transformer_decoder = TransformerDecoder(layer, nlayer)
        self.head = nn.Linear(dim, 2*dim)

    def KL(self, posterior):
        # prior is standard gaussian distribution
        mu, logsigma = posterior['mu'], posterior['logsigma']
        # no matter what shape
        logvar = logsigma*2
        loss = 0.5 * torch.sum(mu * mu+ torch.exp(logvar) - 1 - logvar, 1)
        return loss

    def forward(self, src, src_mask, tgt, tgt_mask):
        """
        src, tgt [L, b, dim]
        src_mask, tgt_mask, [B, L]
        """
        l, b, dim = src.shape
        src_mask, tgt_mask = src_mask.permute(0, 1), tgt_mask.permute(0, 1)
        decoder_output = self.transformer_decoder(src, tgt,
                                                  memory_key_padding_mask=tgt_mask, tgt_key_padding_mask=src_mask).permute(1, 2, 0)
        # [L, B, dim] to [B, dim, L]
        tmp = decoder_output * (1-src_mask.float().unsqueeze(1))
        tmp = tmp.mean(dim=2)
        # [B, dim]
        posterior = self.head(tmp)
        result = {}
        result['mu'] = posterior[:, 0:dim]
        result['logsigma'] = posterior[:, dim:]
        return result, self.KL(result)

class TemperatureScaler(nn.Module):
    def __init__(self, dim, nhead, nlayer, dropout):
        super().__init__()
        self.scale = nn.Linear(1, 1)
    def forward(self, temperature):
        # input: [B, 1]
        return torch.sigmoid(self.scale(temperature.float().unsqueeze(-1)))
    
class MoleculeDecoder(nn.Module):
    def __init__(self, vae, dim, nhead, nlayer, dropout, rank=0, args=None):
        super().__init__()
        layer = TransformerEncoderLayer(dim, nhead, dim, dropout)
        self.transformer_encoder = TransformerEncoder(layer, nlayer)
        self.latent_head = nn.Linear(dim, dim)
        self.bond_decoder = BondDecoder(dim, rank)
        self.charge_head = nn.Conv1d(dim, 13, 1) #-6 to +6
        self.aroma_head = nn.Conv1d(dim, 1, 1)
        self.vae = vae
        self.rank = rank
        self.args = args
        self.scale = TemperatureScaler(dim, nhead, nlayer, dropout)
        
    def forward(self, src, src_bond, src_mask, latent, tgt_bond, tgt_aroma, tgt_charge, tgt_mask, action_embedding=None, temps = None, solvent_embedding=None, catalyst_embedding=None):
        l, b, dim = src.size()
        # print(src.shape, solvent_embedding.shape, catalyst_embedding.shape)
        if self.vae:
            tmp = torch.randn(b, dim).to(self.rank)
            latent = tmp * latent['logsigma'].exp() + latent['mu']
            latent = self.latent_head(latent)
            src = src + latent.expand(l, b, dim)
        if action_embedding is not None:
            src_mask = torch.cat([src_mask, torch.zeros(src_mask.size(0), action_embedding.size(0)).bool().to(src_mask.device)], dim=1)
        if temps is not None:
            src = src + self.scale(temps.to(src.device)) * torch.randn(1, b, dim).to(self.rank)
        if solvent_embedding is not None:
            src_mask = torch.cat([src_mask, torch.zeros(src_mask.size(0), solvent_embedding.size(0)).bool().to(src_mask.device)], dim=1)
        if catalyst_embedding is not None:
            src_mask = torch.cat([src_mask, torch.zeros(src_mask.size(0), catalyst_embedding.size(0)).bool().to(src_mask.device)], dim=1)
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        if action_embedding is not None:
            l = l - action_embedding.size(0)
            encoder_output = encoder_output[:l,:,:]
            src_mask = src_mask[:,:l]
        if solvent_embedding is not None:
            l = l - solvent_embedding.size(0)
            encoder_output = encoder_output[:l,:,:]
            src_mask = src_mask[:,:l]
        if catalyst_embedding is not None:
            l = l - catalyst_embedding.size(0)
            encoder_output = encoder_output[:l,:,:]
            src_mask = src_mask[:,:l]
        eps = 1e-6
        result = self.bond_decoder(encoder_output, src_bond, src_mask, tgt_bond, tgt_mask)
        
            
        tgt_mask = 1-tgt_mask.float()
        encoder_output = encoder_output.permute(1, 2, 0)

        aroma_logit = self.aroma_head(encoder_output)
        BCE = nn.BCEWithLogitsLoss(reduction='none')
        tgt_aroma = tgt_aroma.bool().float()
        aroma_logit = aroma_logit.view(b, l)
        aroma_loss = BCE(aroma_logit, tgt_aroma.float()) #[B, L]
        aroma_loss = aroma_loss * tgt_mask
        aroma_loss = aroma_loss.sum(dim=1)
        result['aroma_loss'] = aroma_loss

        charge_logit = self.charge_head(encoder_output)
        CE = nn.CrossEntropyLoss(reduction='none')
        # assumes [B, C, L] (input, target)        
        tgt_charge = tgt_charge.long() + 6
        charge_loss = CE(charge_logit, tgt_charge)
        charge_loss = charge_loss * tgt_mask
        charge_loss = charge_loss.sum(dim=1)
        result['charge_loss'] = charge_loss
        result['pred_loss'] = result['bond_loss'] + aroma_loss + charge_loss
        return result

    def forward_logits(self, src_embedding, src_bond, padding_mask, temperature=1, action_embedding=None, temps = None, solvent_embedding=None, catalyst_embedding=None):
        return self.sample(src_embedding, src_bond, padding_mask, temperature=temperature, action_embedding=action_embedding, temps = temps, solvent_embedding=solvent_embedding, catalyst_embedding=catalyst_embedding, return_logits=True)

    def sample(self, src_embedding, src_bond, padding_mask, temperature=1, action_embedding=None, temps = None, solvent_embedding=None, catalyst_embedding=None, return_logits=False):
        """
            decode the molecule into bond [B, L, 4], given representation of [L, b, dim]
        """
        l, b, dim = src_embedding.shape

        latent = 0
        if self.vae:
            latent = torch.randn(1, b, dim).to(self.rank) *temperature
            latent = self.latent_head(latent)
        if self.args.use_temp:
            
            src_embedding = src_embedding + self.scale(temps.to(src_embedding.device)) * torch.randn(1, b, dim).to(self.rank)
        
        if action_embedding is not None:
            padding_mask = torch.cat([padding_mask, torch.zeros(padding_mask.size(0), action_embedding.size(0)).bool().to(padding_mask.device)], dim=1)
        if solvent_embedding is not None:
            padding_mask = torch.cat([padding_mask, torch.zeros(padding_mask.size(0), solvent_embedding.size(0)).bool().to(padding_mask.device)], dim=1)
        if catalyst_embedding is not None:
            padding_mask = torch.cat([padding_mask, torch.zeros(padding_mask.size(0), catalyst_embedding.size(0)).bool().to(padding_mask.device)], dim=1)
        encoder_output = self.transformer_encoder(src_embedding, src_key_padding_mask=padding_mask)
        if action_embedding is not None:
            l = l - action_embedding.size(0)
            encoder_output = encoder_output[:l,:,:]
            padding_mask = padding_mask[:,:l] 
        if solvent_embedding is not None:
            l = l - solvent_embedding.size(0)
            encoder_output = encoder_output[:l,:,:]
            padding_mask = padding_mask[:,:l]
        if catalyst_embedding is not None:
            l = l - catalyst_embedding.size(0)
            encoder_output = encoder_output[:l,:,:]
            padding_mask = padding_mask[:,:l]
        result = {}
        if return_logits:
            return self.bond_decoder(encoder_output, src_bond, padding_mask, return_logits=True)
        bond = self.bond_decoder(encoder_output, src_bond, padding_mask)
        result['bond'] = bond.long()
        encoder_output = encoder_output.permute(1, 2, 0)
        # to [b, c, l]
        aroma_logit = self.aroma_head(encoder_output)
        aroma = (aroma_logit > 0).view(b, l)
        result['aroma'] = aroma.long()

        charge_logit = self.charge_head(encoder_output)
        charge = torch.argmax(charge_logit, dim=1)- 6
        result['charge'] = charge.long()

        return result


class ActionEncoder(nn.Module):
    def __init__(self, dim, nhead=8, nlayer=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.action_embedding = nn.Embedding(1000, dim)  # 词表大小设为100
        layer = TransformerEncoderLayer(dim, nhead, dim, dropout)
        self.transformer_encoder = TransformerEncoder(layer, nlayer)
        self.mol_encoder = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        def dummy_warn_if_padding_and_no_attention_mask(self, *args, **kwargs):
            pass
        self.mol_encoder.warn_if_padding_and_no_attention_mask = dummy_warn_if_padding_and_no_attention_mask
        self.mol_linear = nn.Linear(768, dim)
    def forward(self, tokenized_actions, molecular_tokens=None):
        # tokenized_actions: [batch_size, seq_len]
        seq_len = tokenized_actions.size(1)
        if molecular_tokens is not None:
            N = len(molecular_tokens)
        else:
            N = 0
            
        # 生成position encoding
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim))
        pos_embedding = torch.zeros(seq_len, self.dim).to(tokenized_actions.device)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        
        # 获取action embedding并添加位置编码
        # [batch_size, seq_len, dim]
        embedding = self.action_embedding(tokenized_actions)
        
        if molecular_tokens is not None and N > 0:
            # 处理每个batch的分子tokens
            all_embeddings = []
            mol_counts = []
            
            for tokens in molecular_tokens:
                # 将当前batch的所有tokens堆叠在一起
                batch_tokens = torch.cat([t if t.dim() == 2 else t.unsqueeze(0) for t in tokens], dim=0)
                
                # 一次性获取这个batch的所有embeddings
                with torch.no_grad():
                    batch_outputs = self.mol_encoder(batch_tokens)
                batch_embeddings = self.mol_linear(batch_outputs.pooler_output)
                
                all_embeddings.append(batch_embeddings)
                mol_counts.append(len(tokens))
            
            # 合并所有batch的embeddings
            all_mol_embeddings = torch.cat(all_embeddings, dim=0)
            mol_counts = torch.tensor(mol_counts, device=tokenized_actions.device)
            
            # 创建batch索引和token索引
            batch_indices = torch.arange(N, device=tokenized_actions.device).repeat_interleave(mol_counts)
            token_indices = torch.cat([torch.arange(count.item(), device=tokenized_actions.device) 
                                     for count in mol_counts])
            
            # 创建一个大的mask矩阵 [batch_size, seq_len, num_mols]
            token_values = token_indices.unsqueeze(0).unsqueeze(0) + 4  # [1, 1, num_mols]
            token_matches = (tokenized_actions.unsqueeze(-1) == token_values)  # [batch_size, seq_len, num_mols]
            batch_mask = torch.arange(N, device=tokenized_actions.device).unsqueeze(-1) == batch_indices
            
            # 使用爱因斯坦求和来更新embedding
            mol_contribution = torch.einsum('bsm,bm,md->bsd',
                                          token_matches.float(),
                                          batch_mask.float(),
                                          all_mol_embeddings)
            
            # 更新embedding，只在有匹配的位置
            has_match = token_matches.any(dim=-1)
            embedding = torch.where(has_match.unsqueeze(-1), mol_contribution, embedding)
        
        # 转换维度顺序以适应transformer
        embedding = embedding.transpose(0, 1)
        
        # 通过transformer
        output = self.transformer_encoder(embedding)
        output = output.mean(dim=0).unsqueeze(0)
        return output


class MoleculeVAE(nn.Module):
    def __init__(self, args, ntoken, dim=128, nlayer=8, nhead=8, dropout=0.1):
        super().__init__()
        self.args = args
        self.rank = args.local_rank
        self.M_encoder = MoleculeEncoder(ntoken, dim, nhead, nlayer, dropout, self.rank, args)
        self.P_encoder = MoleculeEncoder(ntoken, dim, nhead, nlayer, dropout, self.rank, args)
        if args.vae:
            self.V_encoder = VariationalEncoder(dim, nhead, nlayer, dropout, self.rank)
        self.M_decoder = MoleculeDecoder(args.vae, dim, nhead, nlayer, dropout, self.rank, args)
        if args.action:
            self.action_encoder = ActionEncoder(dim, nhead, nlayer, dropout)
        if args.use_solvent or args.use_catalyst:
            self.agent_encoder = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
            def dummy_warn_if_padding_and_no_attention_mask(self, *args, **kwargs):
                pass
            self.agent_encoder.warn_if_padding_and_no_attention_mask = dummy_warn_if_padding_and_no_attention_mask
            self.agent_linear = nn.Linear(768, dim)
            
    # ====== 新增：纯 encoder 接口 ======
    def encode(self, tensors, which: str = "src"):
        """
        对 src / tgt 图分别做 encode，返回:
            encoder_output: [L, B, D]
            mask:           [B, L]

        which = "src" 或 "tgt"
        目前 encode 不管 action / solvent / catalyst，仅用于 FM 和上层调用。
        """
        elem = tensors["element"]
        reactant_mask = tensors.get("reactant", None)

        if which == "src":
            bond = tensors["src_bond"]
            aroma = tensors["src_aroma"]
            charge = tensors["src_charge"]
            mask = tensors["src_mask"]
            segment = tensors["src_segment"]
        elif which == "tgt":
            bond = tensors["tgt_bond"]
            aroma = tensors["tgt_aroma"]
            charge = tensors["tgt_charge"]
            mask = tensors["tgt_mask"]
            segment = tensors["tgt_segment"]
        else:
            raise ValueError(f"encode(which=...) 必须是 'src' 或 'tgt'，现在是 {which}")

        enc = self.M_encoder(
            elem,
            bond,
            aroma,
            charge,
            mask,
            segment,
            reactant_mask,
            action_embedding=None,
            solvent_embedding=None,
            catalyst_embedding=None,
        )
        return enc, mask

    # ====== 新增：纯 decoder 接口 ======
    def decode(
        self,
        src_enc,
        tensors,
        action_embedding=None,
        temps=None,
        solvent_embedding=None,
        catalyst_embedding=None,
    ):
        """
        给定 src encoder 输出 + tensors，跑一次 decoder，返回 result dict：
            包含 bond_loss / aroma_loss / charge_loss / pred_loss / (如果有的话 kl, loss 等)
        注意：这里不走 VAE 分支，只用当前的 M_decoder。
        """
        src_bond = tensors["src_bond"]
        src_mask = tensors["src_mask"]
        tgt_bond = tensors["tgt_bond"]
        tgt_aroma = tensors["tgt_aroma"]
        tgt_charge = tensors["tgt_charge"]
        tgt_mask = tensors["tgt_mask"]

        result = self.M_decoder(
            src_enc,
            src_bond,
            src_mask,
            latent=None,     # 不走 VAE
            tgt_bond=tgt_bond,
            tgt_aroma=tgt_aroma,
            tgt_charge=tgt_charge,
            tgt_mask=tgt_mask,
            action_embedding=action_embedding,
            temps=temps,
            solvent_embedding=solvent_embedding,
            catalyst_embedding=catalyst_embedding,
        )
        # 和 forward(train) 对齐：加上一个 'loss' 字段
        result["loss"] = result["pred_loss"]
        return result
        
    def forward(self, mode, tensors):
        action_embedding = None
        solvent_embedding = None
        catalyst_embedding = None
        temps = None
        if self.args.use_temp:
            temps = tensors['temperature']
        if self.args.action:
            action_embedding = self.action_encoder(
                tensors['tokenized_actions'],
                tensors['reactant_tokens']
            )
        if self.args.use_solvent:
            solvent_embedding = self.agent_encoder(tensors['solvent'])
            solvent_embedding = self.agent_linear(solvent_embedding.pooler_output)
            solvent_embedding = solvent_embedding.unsqueeze(0)
        if self.args.use_catalyst:
            catalyst_embedding = self.agent_encoder(tensors['catalyst'])
            catalyst_embedding = self.agent_linear(catalyst_embedding.pooler_output)
            catalyst_embedding = catalyst_embedding.unsqueeze(0)

        # 公共的 src encoder（和 encode(which="src") 保持一致）
        src = self.M_encoder(
            tensors['element'],
            tensors['src_bond'],
            tensors['src_aroma'],
            tensors['src_charge'],
            tensors['src_mask'],
            tensors['src_segment'],
            tensors['reactant'],
            action_embedding,
            solvent_embedding,
            catalyst_embedding,
        )

        if mode is 'train':
            bond, aroma, charge = tensors['tgt_bond'], tensors['tgt_aroma'], tensors['tgt_charge']
            if self.args.vae:
                # 原始 VAE 分支保持不动
                tgt = self.P_encoder(
                    tensors['element'],
                    bond,
                    aroma,
                    charge,
                    tensors['tgt_mask'],
                    tensors['tgt_segment'],
                )
                posterior, kl = self.V_encoder(src, tensors['src_mask'], tgt, tensors['tgt_mask'])
                result = self.M_decoder(
                    src,
                    tensors['src_bond'],
                    tensors['src_mask'],
                    posterior,
                    bond,
                    aroma,
                    charge,
                    tensors['tgt_mask'],
                )
                result['kl'] = kl
                result['loss'] = result['pred_loss'] + self.args.beta * kl
            else:
                # 非 VAE 分支逻辑 == 新的 decode 接口
                result = self.M_decoder(
                    src,
                    tensors['src_bond'],
                    tensors['src_mask'],
                    latent=None,
                    tgt_bond=bond,
                    tgt_aroma=aroma,
                    tgt_charge=charge,
                    tgt_mask=tensors['tgt_mask'],
                    action_embedding=action_embedding,
                    temps=temps,
                    solvent_embedding=solvent_embedding,
                    catalyst_embedding=catalyst_embedding,
                )
                result['loss'] = result['pred_loss']

        elif mode is 'sample':
            result = self.M_decoder.sample(
                src,
                tensors['src_bond'],
                tensors['src_mask'],
                temperature,
                action_embedding,
                temps,
                solvent_embedding,
                catalyst_embedding,
            )

        return result