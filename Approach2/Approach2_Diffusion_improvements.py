import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import re
import os
import json

from sympy import sympify, preorder_traversal, SympifyError

class FormulaTokenizer:
    def __init__(self):
<<<<<<< HEAD
        self.token_to_id = defaultdict(lambda: self.token_to_id["[UNK]"])
=======
        self.token_to_id = defaultdict(lambda: self.token_to_id["[UNK]"]) 
>>>>>>> 8b9b756b305a8984ce9397944782289fe402237f
        self.token_to_id["[PAD]"] = 0
        self.token_to_id["[UNK]"] = 1
        self.token_to_id["[OPEN_PAREN]"] = 2
        self.token_to_id["[CLOSE_PAREN]"] = 3
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def build_vocab(self, tokenized_formulas):

        for formula in tokenized_formulas:
            for token in formula:
                if token not in self.token_to_id:
                    new_id = len(self.token_to_id)
                    self.token_to_id[token] = new_id
                    self.id_to_token[new_id] = token

    def encode(self, tokens, max_length):

        token_ids = [self.token_to_id[token] for token in tokens]
        if len(token_ids) < max_length:
            token_ids += [self.token_to_id["[PAD]"]] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]
        return token_ids

    def decode(self, token_ids):

        tokens = [self.id_to_token.get(token_id, "[UNK]") for token_id in token_ids]
        return [token for token in tokens if token != "[PAD]"]
    
    def hierarchical_tokenize(self, formula):

        try:
            expr = sympify(formula)  # Parse the formula into a sympy expression
            tokens = [str(term) for term in preorder_traversal(expr)]
            return tokens
        except SympifyError:
            print(f"Error parsing formula: {formula}")
            return ["[UNK]"]
        
    def regex_tokenize(self, formula):

            formula = formula.replace("(", " [OPEN_PAREN] ").replace(")", " [CLOSE_PAREN] ")
            pattern = r"""
                (var_\d+)          |  # Match variables like var_0, var_1
                (C_\d+)            |  # Match constants like C_0, C_1
                ([a-zA-Z_]+)       |  # Match functions like sqrt, sin, log
                (\[OPEN_PAREN\])   |  # Match custom open parenthesis tag
                (\[CLOSE_PAREN\])  |  # Match custom close parenthesis tag
                ([\+\-\*/\^])      |  # Match operators
                (\d+\.\d+|\d+)        # Match numbers (integers or decimals)
            """
            regex = re.compile(pattern, re.VERBOSE)
            tokens = [match.group(0) for match in regex.finditer(formula)]
            return tokens

# Define Formula Embedding Model
class FormulaEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids):
        return self.embedding(token_ids)

# Define Dataset Class for Precomputed Embeddings
class DiffusionDatasetWithTargets(Dataset):
    def __init__(self, precomputed_data):
        self.data = precomputed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Tokenize dataset with hybrid tokenization
def hybrid_tokenize_formula(formula, tokenizer):
    hierarchical_tokens = tokenizer.hierarchical_tokenize(formula)
    final_tokens = []
    for token in hierarchical_tokens:
        refined_tokens = tokenizer.regex_tokenize(token)
        final_tokens.extend(refined_tokens)
    return final_tokens


def precompute_embeddings_with_targets(data, normalized_data, tokenizer, embedding_model, max_length):
    precomputed = []

    for item, norm_item in zip(data, normalized_data):
        # Use hybrid tokenization for consistent preprocessing
        tokens = hybrid_tokenize_formula(item['formula_human_readable'], tokenizer)
        token_ids = torch.tensor(tokenizer.encode(tokens, max_length), dtype=torch.long)

        # Generate embedding
        with torch.no_grad():
            embedding = embedding_model(token_ids.unsqueeze(0)).squeeze(0)

        # Use normalized data directly
        normalized_points = {key: norm_item[key].values for key in norm_item.columns if key != 'target'}
        normalized_target = norm_item['target'].values

        precomputed.append((normalized_points, embedding, normalized_target))

    return precomputed


# Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        # Input projection
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Self-attention layers
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):

        # Project input embeddings to hidden dimensions
        x_hidden = self.embedding_processor(x)  # Shape: [batch_size, seq_len, hidden_dim]

        # Apply self-attention
        attention_out, _ = self.self_attention(x_hidden, x_hidden, x_hidden)  # Shape: [batch_size, seq_len, hidden_dim]

        # Residual connection for self-attention
        x_residual = x_hidden + attention_out

        # Pass through feed-forward network
        output = self.feed_forward(x_residual)  # Shape: [batch_size, seq_len, embedding_dim]
        
        return output
    
# Noise Scheduler with remove_noise
class NoiseScheduler:
    def __init__(self, timesteps, beta_start=1e-4, beta_end=2e-2):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)  # Noise tensor with the same shape as x
        t = t.long()  # Ensure t is of type long for indexing

        # Expand alpha values to match the batch dimension
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().view(-1, 1, 1)  # Match batch size and feature dimensions
        sqrt_one_minus_alpha_cumprod = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1)  # Match dimensions

        # Apply noise
        x_t = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise
    
    def remove_noise(self, x_t, t, model):

        t = t.long()  # Ensure t is of type long for indexing

        # Predict noise using the model
        predicted_noise = model(x_t)

        # Calculate the reverse step with numerical safeguards
        alpha_t = self.alpha_cumprod[t].view(-1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_recip_alpha_t = (1 / alpha_t).sqrt()
        sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_t, min=1e-7))

        x = sqrt_recip_alpha_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_t)

        # Clamp values to prevent numerical explosion
        return torch.clamp(x, min=-1e5, max=1e5)




# Example Usage
if __name__ == "__main__":
    # Define the paths to the folders
    folders = ["data_symbolic_regression/test", "data_symbolic_regression/train", "data_symbolic_regression/val"]

    # Dictionary to hold the data from each folder
    data = {folder: [] for folder in folders}

    # Iterate through each folder and read JSON files
    for folder in folders:
        if os.path.exists(folder):
            for file_name in os.listdir(folder):
                if file_name.endswith(".json"):
                    file_path = os.path.join(folder, file_name)
                    try:
                        with open(file_path, "r") as file:
                            content = json.load(file)
                            data[folder].append(content)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        else:
            print(f"Folder {folder} does not exist.")

    # Extract data
    train = data["data_symbolic_regression/train"]
    val = data["data_symbolic_regression/val"]
    test = data["data_symbolic_regression/test"]

    # Normalize data
    normalized_train = []
    for item in train:
        scaler = MinMaxScaler()
        df = pd.DataFrame.from_dict(item['points'])
        normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        normalized_train.append(normalized_df)
    
    # Normalize data
    normalized_val = []
    for item in train:
        scaler = MinMaxScaler()
        df = pd.DataFrame.from_dict(item['points'])
        normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        normalized_val.append(normalized_df)

    # Tokenizer and embedding model setup
    tokenizer = FormulaTokenizer()
    tokenized_formulas = [re.findall(r'\w+|\S', item['formula_human_readable']) for item in train]
    tokenizer.build_vocab(tokenized_formulas)

    vocab_size = len(tokenizer.token_to_id)
    embedding_dim = 128
    max_length = 20
    embedding_model = FormulaEmbeddingModel(vocab_size, embedding_dim)

    # Precompute embeddings
    precomputed_train = precompute_embeddings_with_targets(train, normalized_train, tokenizer, embedding_model, max_length)
    # Precompute embeddings
    precomputed_val = precompute_embeddings_with_targets(val, normalized_val, tokenizer, embedding_model, max_length)

    # Initialize diffusion model and noise scheduler
    hidden_dim = 256
    diffusion_model = DiffusionModel(embedding_dim, hidden_dim)
    timesteps = 1000
    scheduler = NoiseScheduler(timesteps)

    # Training setup
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(100):
        diffusion_model.train()
        total_loss = 0
        for points, embedding, target in precomputed_train:
            # Add noise
            t = torch.randint(0, timesteps, (embedding.size(0),))
            x_t, noise = scheduler.add_noise(embedding.unsqueeze(0), t)

            # Forward pass through the diffusion model
            pred_noise = diffusion_model(x_t)

            # Compute loss
            loss = criterion(pred_noise, noise)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(precomputed_train):.4f}")
    
    # Save the model after all epochs are completed
    model_path = "diffusion_model_final.pth"
    torch.save(diffusion_model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")


    diffusion_model.eval()
    val_loss = 0
    with torch.no_grad():
        for points, embedding, target in precomputed_val:
            t = torch.randint(0, timesteps, (embedding.size(0),))  # Random timesteps for validation
            x_t, noise = scheduler.add_noise(embedding.unsqueeze(0), t)

            # Forward pass through the diffusion model
            pred_noise = diffusion_model(x_t)  
            loss = criterion(pred_noise, noise)  
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(precomputed_val):.4f}")



    # Get the first training data point
    data_point = precomputed_train[0]
    points, embedding, target = data_point

    # Start denoising
    print("Starting denoising for the first training data point...")
    t = torch.tensor([scheduler.timesteps - 1])  
    x_t = embedding.unsqueeze(0)  

    while t > 0:
        # Expand t to match batch dimensions (if required)
        t_expanded = t.expand(x_t.size(0))

        # Perform a reverse step using the trained model
        x_t = scheduler.remove_noise(x_t, t_expanded, diffusion_model)

        # Decrement timestep
        t -= 1

    # Final denoised embedding
    denoised_embedding = x_t.squeeze(0) 

    
    print("Denoised embedding:", denoised_embedding)

    # Inspect token indices after argmax
    denoised_tokens = denoised_embedding.argmax(dim=-1).tolist()
    print("Denoised token indices:", denoised_tokens)

    # Decode tokens back to formula
    formula = tokenizer.decode(denoised_tokens)

    print("Regenerated formula:", formula)

        