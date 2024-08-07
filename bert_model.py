import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BertConfig():
    max_seq_len: int = 512
    vocab_size: int = 30522
    n_layers: int = 12
    n_heads: Tuple[int] = (12, ) * n_layers
    emb_size: int = 768
    intermediate_size: int = 4 * emb_size
    dropout: float = 0.1
    n_classes: int = 2
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    return_pooler_output: bool = False
   

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.emb_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.emb_size)
        self.token_type_embeddings = nn.Embedding(2, config.emb_size)
        self.LayerNorm = nn.LayerNorm(config.emb_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position_ids", torch.arange(config.max_seq_len).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
    
    def forward(self, input_ids, token_type_ids, 
                position_ids = None, inputs_embeds = None):
        input_shape = input_ids.shape
        
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:,0 : seq_length ]
        
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            word_emb = self.word_embeddings(input_ids)
        else:
            word_emb = inputs_embeds
        token_emb = self.token_type_embeddings(token_type_ids)
        position_emb = self.position_embeddings(position_ids)
    
        emb = word_emb + token_emb + position_emb    
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)
        
        return emb
    
class BertSelfAttention(nn.Module):
    def __init__(self, config, layer_i):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads[layer_i]
        self.head_size = config.emb_size // self.n_heads
        self.query = nn.Linear(config.emb_size, config.emb_size)
        self.key = nn.Linear(config.emb_size, config.emb_size)
        self.value = nn.Linear(config.emb_size, config.emb_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, emb, att_mask):
        B, T, C = emb.shape  # batch size, sequence length, embedding size   
    
        q = self.query(emb).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = self.key(emb).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(emb).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        
        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5

        # set the pad tokens to -inf so that they equal zero after softmax
        if att_mask != None:
            att_mask = (att_mask > 0).unsqueeze(1).repeat(1, att_mask.size(1), 1).unsqueeze(1)
            weights = weights.masked_fill(att_mask == 0, float('-inf'))

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        emb_rich = weights @ v
        emb_rich = emb_rich.transpose(1, 2).contiguous().view(B, T, C)   
        return emb_rich
    
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.emb_size, config.emb_size)
        self.LayerNorm = nn.LayerNorm(config.emb_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, emb_rich, emb):
        x = self.dense(emb_rich)
        x = self.dropout(x)
        x = x + emb
        out = self.LayerNorm(x)
        return out
  
class BertAttention(nn.Module):
    def __init__(self, config, layer_i):
        super().__init__()
        self.self = BertSelfAttention(config, layer_i)
        self.output = BertSelfOutput(config)

    def forward(self, emb, att_mask):
        emb_rich = self.self(emb, att_mask)
        out = self.output(emb_rich, emb)
        return out

    
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.emb_size, config.intermediate_size)
        self.gelu = nn.GELU() 

    def forward(self, att_out):
        x = self.dense(att_out)
        out = self.gelu(x)
        return out
    
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.emb_size)
        self.LayerNorm = nn.LayerNorm(config.emb_size, eps=config.layer_norm_eps) 
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, intermediate_out, att_out):
        x = self.dense(intermediate_out)
        x = self.dropout(x)
        x = x + att_out
        out = self.LayerNorm(x)
        return out 

class BertLayer(nn.Module):
    def __init__(self, config, layer_i):
        super().__init__()
        self.attention = BertAttention(config, layer_i)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config) 

    def forward(self, emb, att_mask):
        att_out = self.attention(emb, att_mask)
        intermediate_out = self.intermediate(att_out)
        out = self.output(intermediate_out, att_out)
        return out
        
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config, layer_i) for layer_i in range(config.n_layers)])
    
    def forward(self, emb, att_mask):
        for bert_layer in self.layer:
            emb = bert_layer(emb, att_mask)
        return emb

    
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.emb_size, config.emb_size)
        self.tanh = nn.Tanh()

    def forward(self, encoder_out):
        pool_first_token = encoder_out[:, 0]
        out = self.dense(pool_first_token)
        out = self.tanh(out)
        return out
    
class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids, att_mask):
        emb = self.embeddings(input_ids, token_type_ids)
        out = self.encoder(emb, att_mask)
        pooled_out = self.pooler(out)
        return out, pooled_out
    
class BertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.emb_size, config.n_classes)

        
    def forward(self, input_ids, token_type_ids, attention_mask=None):
        _, pooled_out = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_out = self.dropout(pooled_out)
        logits = self.classifier(pooled_out)
        
        if self.config.return_pooler_output:
            return pooled_out, logits        
        return logits
     
    def reduce_seq_len(self, seq_len):
       
        assert seq_len <= self.config.max_seq_length, f"Sequence length must be reduced below current \
            length of {self.config.max_seq_length}"
        self.bert.embeddings.position_embeddings.weight = nn.Parameter(self.bert.embeddings.position_embeddings.weight[:seq_len])
        self.bert.embeddings.position_ids = self.bert.embeddings.position_ids[:, :seq_len]
        self.config.max_seq_length = seq_len
        
    @staticmethod
    def adaptive_copy(orig_wei, new_wei):       
                        
        n_dim = orig_wei.dim()
        with torch.no_grad():
            if n_dim == 1:
                dim1 = list(orig_wei.shape)[0]
                orig_wei.copy_(new_wei[:dim1])
            elif n_dim == 2:
                dim1, dim2 = list(orig_wei.shape)
                orig_wei.copy_(new_wei[:dim1, :dim2])
            elif n_dim == 3:
                dim1, dim2, dim3 = list(orig_wei.shape)
                orig_wei.copy_(new_wei[:dim1, :dim2, :dim3])
        
    @classmethod
    def from_pretrained(cls, model_type, config_args=None, adaptive_weight_copy=False):
                         
        from transformers import BertForSequenceClassification as HFBertForSequenceClassification
        
        print(f"Loading weights from pretrained model: {model_type}")
        
        if config_args:
            config = BertConfig(**config_args)
        else:
            config = BertConfig()
        
        model = cls(config)        
        sd = model.state_dict()
        sd_keys = sd.keys()
        
        # init huggingface/transformers model
        model_hf = HFBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_type, num_labels=config.n_classes)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()

        
        failed_key = []
        
        for k in sd_keys:
            if k not in sd_keys_hf:
                failed_key.append(k)

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}. {failed_key}"
            
        
        
        for k in sd_keys_hf:
            
            if not adaptive_weight_copy:
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch: {k} --> (hf vs custom) ({sd_hf[k].shape} vs {sd[k].shape})"  
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
            else:
                with torch.no_grad():
                    cls.adaptive_copy(sd[k], sd_hf[k])                  
        return model
