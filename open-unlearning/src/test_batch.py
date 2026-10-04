from __future__ import annotations
import os
import re
import math
import copy
import logging
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data.qa import QADataset
from data.pretraining import PretrainingDataset, CompletionDataset
from data.utils import load_hf_dataset, preprocess_pretraining_instance
from data.collators import DataCollatorForSupervisedDataset
from evals.metrics.utils import evaluate_probability, eval_text_similarity, tokenwise_vocab_logprobs


logger = logging.getLogger("EvaluatorComputeSampleDifficulty")


def sentence_split(text):
    '''Source: https://stackoverflow.com/questions/4576077/python-split-text-on-sentences'''
    
    digits = "([0-9])"
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def split_sentence_and_flatten(raw_texts):
    sentences = []
    for text in raw_texts:
        sents = sentence_split(text)
        sentences.extend(sents)

    return sentences

class SentenceDataset(Dataset):
    def __init__(
        self, hf_args, tokenizer, text_key="text", max_length=256
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if 'TOFU' in hf_args['path']:
            self.raw_data = load_hf_dataset(**hf_args)
            self.raw_data = self.raw_data.map(lambda x: {text_key: x['question'] + ' ' + x['answer']})[text_key]

        else:
            self.raw_data = load_hf_dataset(**hf_args)[text_key]
        
        self.data = split_sentence_and_flatten(self.raw_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.data[idx], self.max_length
        )

def prob(model, tokenizer, batch):
    res = evaluate_probability(model, batch)
    return [i['prob'] for i in res]

def loss(model, tokenizer, batch):
    res = evaluate_probability(model, batch)
    return [i['avg_loss'] for i in res]

def rougeL(model, tokenizer, batch):
    args = OmegaConf.create(
        {'do_sample': False, 'top_p': None, 'temperature': None, 'max_new_tokens': 32, 
         'use_cache': True, 'stopwords': ['\n\n', '\nQuestion', 'Question:']})
    res = eval_text_similarity(
        model, 
        tokenizer, 
        batch, 
        generation_args=args
    )
    return [i['rougeL_recall'] for i in res]

def exact_mem(model, tokenizer, batch):
    log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
        model, batch, grad=False, return_labels=True
    )
    em_batch = []
    for log_probs, labels in zip(log_probs_batch, labels_batch):
        valid_len = len(labels)
        if valid_len == 0:
            # Rarely, tokenization can result in a mismatch with no valid target
            # tokens for loss computation (see preprocess_chat_instance() for
            # reference). Since this condition makes no sense in terms of
            # computing EM, we just choose to set EM=None
            logger.warning(
                "EM score for an instance is marked None, due to "
                "tokenization issues that resulted in no valid target tokens."
            )
            # em_batch.append({"score": None})
            em_batch.append({"score": 0})
        else:
            preds = torch.argmax(log_probs, dim=-1)
            em_score = (preds == labels).sum() / valid_len
            em_batch.append({"score": em_score.item()})

    return [i['score'] for i in em_batch]

def extraction_strength(model, tokenizer, batch):
    log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
        model, batch, grad=False, return_labels=True
    )
    es_batch = []
    for log_probs, labels in zip(log_probs_batch, labels_batch):
        valid_len = len(labels)
        preds = torch.argmax(log_probs, dim=-1)
        for k in range(valid_len):
            suff_preds = preds[k:]
            suff_labels = labels[k:]
            if torch.equal(suff_preds, suff_labels):
                break
        if valid_len == 0:
            # Rarely, tokenization can result in a mismatch with no valid target
            # tokens for loss computation (see preprocess_chat_instance() for
            # reference). Since this condition makes no sense in terms of
            # computing ES, we just choose to set ES=None
            logger.warning(
                "ES score for an instance is marked None, due to "
                "tokenization issues that resulted in no valid target tokens."
            )
            es_batch.append({"score": 0})
        else:
            es_score = 1 - (k / valid_len)
            es_batch.append({"score": es_score})
    
    return [i['score'] for i in es_batch]

def compute_MRD(Pt, Pt_perturb, valid_mask, eps=0.0):
    if Pt.shape != Pt_perturb.shape:
        raise ValueError(f"Pt and Pt_perturb must have same shape, got {Pt.shape} vs {Pt_perturb.shape}")

    # tokens to include: Pt != 0 (and valid_mask if provided)
    nonzero = Pt != 0
    if valid_mask.shape != Pt.shape:
        raise ValueError(f"valid_mask must match Pt shape, got {valid_mask.shape} vs {Pt.shape}")
    mask = nonzero & valid_mask
    
    ratio = torch.zeros_like(Pt, dtype=torch.float32)
    ratio = ratio.where(~mask, (Pt - Pt_perturb) / Pt)  # fill only masked-true positions

    # sum over token dimension
    ratio = ratio.masked_fill(~mask, 0.0)
    return ratio.sum(dim=1)


def _get_token_probabilities(model,input_ids,tokenizer):
    input_ids = input_ids.squeeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    probabilities = []
    res = 0
    with torch.no_grad():
        outputs = model(input_ids=input_ids.unsqueeze(0))  # 增加batch维度
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

        probs = F.softmax(logits, dim=-1)
        
        for i in range(1, len(input_ids)):
            current_token_id = input_ids[i].item()
            if current_token_id==2:
                break
            token_probs = probs[0, i-1]
            token_prob = token_probs[current_token_id].item()
            
            current_token = tokenizer.decode([current_token_id])
            if token_prob<=0:
                token_prob = 0.00000000000001
            prefix = tokenizer.decode(input_ids[:i])
            probabilities.append({
                "prefix": prefix,
                "token": current_token,
                "probability": token_prob
            })
            res += token_prob
    
    return res/len(probabilities),probabilities

@torch.no_grad()
def get_token_probabilities(model, batch, eos_token_id, min_prob=1e-14):
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}

    B, T = batch['input_ids'].shape
    outputs = model(**batch)
    logits = outputs.logits                       # (B, T, V)
    probs = F.softmax(logits, dim=-1)             # (B, T, V)

    next_probs = probs[:, :-1, :]                 # (B, T-1, V)
    gold_next_ids = batch['input_ids'][:, 1:]     # (B, T-1)

    # ---- gather gold probabilities ----
    token_prob = next_probs.gather(
        dim=-1,
        index=gold_next_ids.unsqueeze(-1)         # (B, T-1, 1)
    ).squeeze(-1)                                 # (B, T-1)

    token_prob = token_prob.clamp_min(min_prob)

    valid_mask = torch.ones((B, T - 1), device=device, dtype=torch.bool)

    if eos_token_id is not None:
        # mask out positions at/after first EOS in gold_next_ids
        eos_hits = (gold_next_ids == eos_token_id)            # (B, T-1)
        # first_eos_idx: 0..T-2, or (T-1) if no eos
        has_eos = eos_hits.any(dim=1)
        first_eos_idx = torch.where(
            has_eos,
            eos_hits.float().argmax(dim=1),
            torch.full((B,), T - 1, device=device, dtype=torch.long),
        )
        arange = torch.arange(T - 1, device=device).unsqueeze(0)  # (1, T-1)
        valid_mask &= (arange < first_eos_idx.unsqueeze(1))

    if batch['attention_mask'] is not None:
        # We are scoring gold_next_ids which correspond to input positions 1..T-1
        valid_mask &= batch['attention_mask'][:, 1:].bool()

    # ---- average per sample safely ----
    denom = valid_mask.sum(dim=1).clamp_min(1)                 # (B,)
    avg_prob = (token_prob * valid_mask).sum(dim=1) / denom    # (B,)
    token_log_prob = token_prob.log()    

    return avg_prob, token_log_prob, valid_mask


def perturb_model_parameters(model, mean=0.0, std=1e-5):
    with torch.no_grad(): 
        for name, param in model.named_parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * std + mean
                param.add_(noise)

def compute_mrd(model, tokenizer, batch, **kwargs):
    '''Adapted from MRD repo: https://github.com/QDRhhhh/MRD/blob/main/src/unlearn/base.py#L137'''
    K = 100
    seed = 42
    MRD = 0
    if 'original_state' in kwargs:
        original_state = kwargs['original_state']
    else:
        original_state = copy.deepcopy(model.state_dict())
    for i in range(0,K):
        torch.manual_seed(seed+i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed+i)
        model.load_state_dict(original_state)
        _, Pt, valid_mask = get_token_probabilities(model, batch, tokenizer.eos_token_id)
        perturb_model_parameters(model)
        _, Pt_perturb, valid_mask = get_token_probabilities(model, batch, tokenizer.eos_token_id)
        MRD += compute_MRD(Pt,Pt_perturb, valid_mask)
    
    MRD = min(abs(MRD/K),2)

    return MRD

class EvaluatorComputeSampleDifficulty:
    """Evaluator that computes a difficulty metric over a forget set."""

    def __init__(self, dataset_name, data_split, cfg, **kwargs): 
        self.dataset_name = dataset_name
        self.data_split = data_split
        self.eval_cfg = cfg
        self.template_args = kwargs['template_args']
        self.tokenizer = kwargs['tokenizer']
        self.collator = None
        self.metrics = None

        # if dataset_name == 'tofu':
        #     hf_args = {"name": self.data_split, 'path': 'locuslab/TOFU', 'split': 'train'}
        # elif dataset_name == 'muse':
        #     hf_args = {"name": 'raw', 'path': f'muse-bench/MUSE-{self.data_split}', 'split': 'forget'}
        # elif dataset_name == 'wmdp':
        #     hf_args = {'path': 'text', 'data_files': f'data/wmdp/wmdp-corpora/{self.data_split}-forget-corpus.jsonl', 'split': 'train'}

        # dataset = SentenceDataset(hf_args, self.tokenizer)
        # self.collator = DataCollatorForSupervisedDataset(self.tokenizer, padding_side="left",)

        if dataset_name == 'tofu':
            hf_args = {"name": self.data_split, 'path': 'locuslab/TOFU', 'split': 'train'}
            dataset = QADataset(
                hf_args, self.template_args, self.tokenizer, max_length=512, predict_with_generate=False)
            self.collator = DataCollatorForSupervisedDataset(self.tokenizer, padding_side="right", index="index")

        elif dataset_name == 'muse':
            max_length = 2048
            hf_args = {"name": 'raw', 'path': f'muse-bench/MUSE-{self.data_split}', 'split': 'forget'}
            # dataset = CompletionDataset(
            #     hf_args, self.template_args, self.tokenizer, max_length=max_length, predict_with_generate=False, insert_space=True)
            dataset = PretrainingDataset(
                hf_args, self.template_args, self.tokenizer, max_length=max_length)
            self.collator = DataCollatorForSupervisedDataset(self.tokenizer, padding_side="left")

        elif dataset_name == 'wmdp':
            hf_args = {'path': 'text', 'data_files': f'data/wmdp/wmdp-corpora/{self.data_split}-forget-corpus.jsonl', 'split': 'train'}
            dataset = PretrainingDataset(
                hf_args, self.template_args, self.tokenizer, max_length=512)
            self.collator = DataCollatorForSupervisedDataset(self.tokenizer, padding_side="left",)


        self.metrics = {
            # 'loss': (loss, dataset), 
            # 'prob': (prob, dataset),
            # 'rouge': (rougeL, dataset_gen),
            # 'exact_mem': (exact_mem, dataset),
            # 'extraction_strength': (extraction_strength, dataset),
            'mrd': (compute_mrd, dataset), 
        }
        for name, metric in self.metrics.items():
            self.metrics[name][0].collators = self.collator

    def prepare_model(self, model):
        """Prepare model for evaluation"""
        model.eval()
        return model

    def compute_sample_difficulty(self, model, tokenizer, output_dir=None, **kwargs):
        # output_dir = os.path.join(output_dir, self.dataset_name, self.data_split, self.trainer)

        # Prepare model for evaluation
        model = self.prepare_model(model)

        logger.info(f"***** Computing forget set sample difficulty *****")
        for metric_name, (metric_fn, dataset) in self.metrics.items():
            path = os.path.join(output_dir, f'{metric_name}.pt')
            # if os.path.exists(path):
            #     continue

            if metric_name == 'mrd':
                kwargs = {'original_state': copy.deepcopy(model.state_dict())}

            dataloader = DataLoader(dataset, batch_size=32, collate_fn=self.collator, shuffle=False)
            evals = []
            for batch in tqdm(dataloader, desc=f'{metric_name}', total=len(dataloader)):
                if 'index' in batch:
                    batch.pop('index')
                batch_out = metric_fn(
                    model=model, tokenizer=tokenizer, batch=batch, **kwargs
                )
                evals.extend(batch_out)
            print("Evaluated", len(evals), "examples")

            res = torch.tensor(evals)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(res, path)
