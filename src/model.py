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
        self._freeze_sentence_transformer_except_last_layers(unfreeze_last)

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

    def _freeze_sentence_transformer_except_last_layers(self, unfreeze_last):
        for parameter in self.sbert.parameters():
            parameter.requires_grad_(False)

        transformer_module = self.sbert[0]
        encoder_layers = transformer_module.auto_model.encoder.layer
        for layer in encoder_layers[-unfreeze_last:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)

        for parameter in self.sbert[1].parameters():
            parameter.requires_grad_(True)

    def _encode_sentence_batch(self, sentence_batch, device):
        transformer_module = self.sbert[0]
        max_sequence_length = transformer_module.max_seq_length or 128
        tokenized_batch = transformer_module.tokenizer(
            sentence_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_sequence_length,
        )
        tokenized_batch = {key: value.to(device) for key, value in tokenized_batch.items()}
        token_embeddings = transformer_module.auto_model(**tokenized_batch).last_hidden_state
        token_mask = tokenized_batch["attention_mask"].unsqueeze(-1).float()
        pooled_embeddings = (token_embeddings * token_mask).sum(1) / token_mask.sum(1).clamp(min=1e-9)
        return pooled_embeddings

    def encode_sentences(self, sentences, device):
        pooled_sentence_embeddings = []
        for start_index in range(0, len(sentences), self.sbert_chunk):
            sentence_batch = sentences[start_index:start_index + self.sbert_chunk]
            pooled_sentence_embeddings.append(self._encode_sentence_batch(sentence_batch, device))
        return torch.cat(pooled_sentence_embeddings, dim=0)

    def _pack_sentence_embeddings_by_transcript(self, projected_embeddings, sentence_counts, device):
        batch_size = sentence_counts.size(0)
        max_sentences = int(sentence_counts.max().item())

        padded_embeddings = projected_embeddings.new_zeros(batch_size, max_sentences, projected_embeddings.size(-1))
        valid_sentence_mask = torch.zeros(batch_size, max_sentences, dtype=torch.bool, device=device)

        offset = 0
        for transcript_index, sentence_count in enumerate(sentence_counts.tolist()):
            sentence_count = int(sentence_count)
            padded_embeddings[transcript_index, :sentence_count] = projected_embeddings[offset:offset + sentence_count]
            valid_sentence_mask[transcript_index, :sentence_count] = True
            offset += sentence_count

        packed_embeddings = pack_padded_sequence(
            padded_embeddings,
            sentence_counts.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        return packed_embeddings, valid_sentence_mask

    def forward(self, sentences, sent_counts, financial, return_attention=False):
        device = financial.device

        sentence_embeddings = self.encode_sentences(sentences, device)
        projected_embeddings = self.proj(sentence_embeddings)

        packed_embeddings, sentence_mask = self._pack_sentence_embeddings_by_transcript(
            projected_embeddings,
            sent_counts,
            device,
        )
        lstm_output, _ = self.bilstm(packed_embeddings)
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        if return_attention:
            transcript_context, attention_weights = self.attention(lstm_output, sentence_mask, return_weights=True)
        else:
            transcript_context = self.attention(lstm_output, sentence_mask)

        financial_features = financial.nan_to_num(0.0)
        logits = self.classifier(self.mlp(torch.cat([transcript_context, financial_features], dim=1)))
        if return_attention:
            return logits, attention_weights
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
        "financial": torch.tensor(financial, dtype=torch.float),
    }
