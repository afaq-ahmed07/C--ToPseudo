import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
from model import Transformer

def simple_tokenizer(text):
    # A simple whitespace and punctuation based tokenizer
    tokens = re.findall(r"[\w]+|[^\s\w]", text)
    return tokens

@st.cache_resource
def load_vocabs():
    with open("src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)
    return src_vocab, tgt_vocab

@st.cache_resource
def load_model(model_path, src_vocab_size, tgt_vocab_size, device):
    model = Transformer(src_vocab_size, tgt_vocab_size, embed_size=128, num_layers=2,
                        num_heads=2, forward_expansion=4, dropout=0.1, max_len=100)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

import torch

def generate_pseudocode(model, code, src_vocab, tgt_vocab, max_length=100, device='cpu'):
    """
    Generates C++ code from input code using a trained Transformer model.

    Args:
        model: The trained Transformer model.
        code (str): The input code string.
        src_vocab (dict): Source vocabulary mapping tokens to indices.
        tgt_vocab (dict): Target vocabulary mapping tokens to indices.
        max_length (int): Maximum number of tokens to generate.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        generated_code (str): The generated C++ code.
    """
    # Ensure the model is on the correct device.
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Tokenize the code and convert tokens to indices using the source vocabulary.
        src_tokens = simple_tokenizer(str(code))
        src_indices = [src_vocab.get(token, src_vocab.get('<unk>')) for token in src_tokens]

        # Convert to tensor with shape [1, src_seq_len] and send to device.
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)

        # Pass through the encoder.
        encoder_output = model.encode(src_tensor, src_mask)

        # Initialize target sequence with the start-of-sequence token.
        sos_token = tgt_vocab.get('<sos>')
        eos_token = tgt_vocab.get('<eos>')
        generated_tokens = [sos_token]

        # Greedy decoding loop.
        for _ in range(max_length):
            # Create target tensor from generated tokens so far.
            tgt_tensor = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(device)
            tgt_mask = model.make_tgt_mask(tgt_tensor)

            # Decode using encoder output and current target sequence.
            decoder_output = model.decode(tgt_tensor, encoder_output, src_mask, tgt_mask)

            # Pass through the final linear layer to get logits over the target vocabulary.
            logits = model.fc_out(decoder_output)  # shape: [1, seq_len, tgt_vocab_size]
            # Focus on the logits of the last token.
            next_token_logits = logits[:, -1, :]  # shape: [1, tgt_vocab_size]
            # Greedy decoding: select the token with the highest logit.
            next_token = torch.argmax(next_token_logits, dim=-1).item()

            generated_tokens.append(next_token)

            # Stop if the end-of-sequence token is generated.
            if next_token == eos_token:
                break

        # Build a reverse mapping from indices to tokens for the target vocabulary.
        rev_tgt_vocab = {idx: token for token, idx in tgt_vocab.items()}
        generated_token_list = [rev_tgt_vocab.get(idx, '<unk>') for idx in generated_tokens]

        # Remove the start token and tokens after the end token.
        if generated_token_list[0] == '<sos>':
            generated_token_list = generated_token_list[1:]
        if '<eos>' in generated_token_list:
            eos_index = generated_token_list.index('<eos>')
            generated_token_list = generated_token_list[:eos_index]

        # Join tokens into a string (adjust spacing/formatting as needed).
        generated_code = ' '.join(generated_token_list)

    return generated_code

st.title("C++ to Psuedocode Generator")

# Load vocabularies and model (cached for efficiency)
src_vocab, tgt_vocab = load_vocabs()
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("model.pth", src_vocab_size, tgt_vocab_size, device)

# User input for code
code_input = st.text_area("Enter code:", "")

if st.button("Generate C++ Code"):
    if code_input:
        generated_pc = generate_pseudocode(model, code_input, src_vocab, tgt_vocab, max_length=100, device=device)
        st.subheader("Generated Pseudo Code:")
        st.text(generated_pc)
    else:
        st.warning("Please enter some code.")
