import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sentence_transformers import SentenceTransformer


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask, return_weights=False):
        s = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=1)
        ctx = (h * w.unsqueeze(-1)).sum(dim=1)
        if return_weights:
            return ctx, w
        return ctx


class TranscriptModel(nn.Module):
    def __init__(self, num_classes=3, sbert_name="all-MiniLM-L6-v2",
                 unfreeze_last=2, sbert_chunk=256,
                 lstm_hidden=256, lstm_layers=2,
                 num_financial=3, dropout=0.3):
        super().__init__()
        self.sbert = SentenceTransformer(sbert_name, device="cpu")
        for p in self.sbert.parameters():
            p.requires_grad_(False)
        encoder = self.sbert[0].auto_model.encoder
        for layer in encoder.layer[-unfreeze_last:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in self.sbert[1].parameters():
            p.requires_grad_(True)

        d = self.sbert.get_embedding_dimension()
        self.sbert_chunk = sbert_chunk

        self.proj = nn.Sequential(
            nn.Linear(d, lstm_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            lstm_hidden, lstm_hidden, lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        d_lstm = lstm_hidden * 2
        self.attention = Attention(d_lstm)
        self.mlp = nn.Sequential(
            nn.Linear(d_lstm + num_financial, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(64, num_classes)

    def sbert_params(self):
        return [p for p in self.sbert.parameters() if p.requires_grad]

    def other_params(self):
        sbert_ids = {id(p) for p in self.sbert.parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in sbert_ids]

    def encode_sentences(self, sentences, device):
        out = []
        tm = self.sbert[0]
        max_len = tm.max_seq_length or 128
        for i in range(0, len(sentences), self.sbert_chunk):
            batch = sentences[i:i + self.sbert_chunk]
            enc = tm.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            o = tm.auto_model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append((o * mask).sum(1) / mask.sum(1).clamp(min=1e-9))
        return torch.cat(out, dim=0)

    def forward(self, sentences, sent_counts, financial, return_attention=False):
        device = financial.device
        B = sent_counts.size(0)

        embs = self.encode_sentences(sentences, device)
        proj = self.proj(embs)

        max_s = int(sent_counts.max().item())
        padded = proj.new_zeros(B, max_s, proj.size(-1))
        mask = torch.zeros(B, max_s, dtype=torch.bool, device=device)
        idx = 0
        for i, n in enumerate(sent_counts.tolist()):
            n = int(n)
            padded[i, :n] = proj[idx:idx + n]
            mask[i, :n] = True
            idx += n

        packed = pack_padded_sequence(padded, sent_counts.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)

        if return_attention:
            ctx, w = self.attention(out, mask, return_weights=True)
        else:
            ctx = self.attention(out, mask)

        fin = financial.nan_to_num(0.0)
        logits = self.classifier(self.mlp(torch.cat([ctx, fin], dim=1)))
        if return_attention:
            return logits, w
        return logits


def collate(batch):
    sentences, sent_counts, labels, financial = [], [], [], []
    from data import split_sentences
    for r in batch:
        s = split_sentences(r["text"])
        if not s:
            s = [r["text"][:200]]
        sentences.extend(s)
        sent_counts.append(len(s))
        labels.append(r["label"])
        financial.append(r["financial"])
    return {
        "sentences": sentences,
        "sent_counts": torch.tensor(sent_counts, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "financial":torch.tensor(financial, dtype=torch.float),
    }
