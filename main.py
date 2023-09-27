import torch
import torch.nn as nn
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load('/Users/plvenard/Downloads/mistral-7B-v0.1/tokenizer.model')

vocab_size = tokenizer.GetPieceSize()
print("Vocab size: ", vocab_size)

class CustomTransformer(nn.Module):
    def __init__(self):
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=512)
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
        self.fc = nn.Linear(512, 10000)

    def forward(self, src, tgt):
        src, tgt = self.embedding(src), self.embedding(tgt)
        output = self.transformer(src, tgt)
        return self.fc(output)

####################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, head_dim, hidden_dim, n_heads):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        
        # Simple feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        # Assuming x is of shape [seq_len, batch_size, dim]
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(attn_output + x)
        ff_output = self.feed_forward(x)
        x = self.norm2(ff_output + x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, head_dim, hidden_dim, n_heads):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, head_dim, hidden_dim, n_heads) for _ in range(n_layers)
        ])
        
        self.output_layer = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # Assuming x is of shape [seq_len, batch_size]
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x
################################   
model = CustomTransformer()
#model = TransformerModel(vocab_size=32000,
    dim=4096,
    n_layers=32,
    head_dim=128,
    hidden_dim=14336,
    n_heads=32
)

weights = torch.load("/Users/plvenard/Downloads/mistral-7B-v0.1/consolidated.00.pth")

#for key, value in weights.items():
    #print(key, value.shape)
print("Loading the model")
model.load_state_dict(weights, strict=False)


# Tokenize text
print("Tokenize text")
tokens = tokenizer.EncodeAsIds('Hello can you help me?')

# Convert tokens to tensor and pass through model
print("Convert tokens to tensor and pass through model")
input_tensor = torch.tensor(tokens).unsqueeze(1)  # Adding a batch dimension
model.eval()
output = model(input_tensor, input_tensor)

# Optionally, detokenize output if necessary
print("Detokenize output ")
text = tokenizer.DecodeIds(output.argmax(dim=-1).squeeze().tolist())
print(text)