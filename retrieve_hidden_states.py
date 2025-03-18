import yaml
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from pathlib import Path
from collections import Counter

from tqdm import tqdm

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_dataset(dataset_path):
    """Load examples from JSONL file."""
    data = []
    with open(dataset_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    max_new_tokens = 3

    # Load model and tokenizer
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with quantization if specified
    model_kwargs = {}
    if config.get('quantization'):
        model_kwargs['device_map'] = 'auto'
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Options: 'nf4' (default), 'fp4'
            bnb_4bit_use_double_quant=True,  # Enable nested quantization
            bnb_4bit_compute_dtype=torch.bfloat16  # Compute in bfloat16 for faster inference
        )
    
    # Set generation config
    model_kwargs["generation_config"] = GenerationConfig(
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_logits=True,
        max_new_tokens=max_new_tokens,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model.eval()

    # Move model to device
    device = config["device"]
    if not config.get('quantization'):  # Don't move if using quantization
        model = model.to(device)
    
    # Load dataset
    dataset = load_dataset(config['dataset_path'])
    
    # Prepare storage for results
    hidden_states = []
    logits = []
    
    # Process each example
    n_test_passes = 3
    output_tok_counter = Counter()
    output_idx = None
    for i, example in enumerate(tqdm(dataset)):
        # Format the prompt with the example text
        if "prompt_template" in config:
            template = config['prompt_template']
        elif "template_path" in config:
            with open(config['template_path']) as f:
                template = f.read()
        else:
            raise ValueError("Either 'prompt_template' or 'template_path' must be specified in the config.")
    
        formatted_prompt = template.format(text=example['text']).strip()
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors='pt')
        inputs = inputs.to(device)
        
        # TODO: support modernbert (should be easier)
        outputs = model.generate(**inputs)

        if i < n_test_passes:
            # We don't know which token index stores the answer yet, so we track where
            # answers appear in the output tokens
            output_seq = outputs.sequences[0][-max_new_tokens:]
            output_toks = tokenizer.convert_ids_to_tokens(output_seq)
            output_tok_counter.update([
                i for i, id in enumerate(output_toks)
                if any(a.startswith(id) for a in config["valid_answers"])
            ])
            
            # Get the output index for all tokens
            final_hidden = [o[-1][:, -1, :].cpu().numpy() for o in outputs.hidden_states]
            next_token_logits = [o.squeeze().cpu().numpy() for o in outputs.logits]

            hidden_states.append(final_hidden)
            logits.append(next_token_logits)
        else:
            if i == n_test_passes:
                # Get the output index for the last token
                if len(output_tok_counter) > 1:
                    print("Warning: multiple output tokens detected. Using the most common one.")
                output_idx = output_tok_counter.most_common(1)[0][0]
                print("Output index:", output_idx)

            # Get final hidden state for the last token
            final_hidden = outputs.hidden_states[output_idx][-1][:, -1, :].cpu().numpy()
            
            # Get logits for the next token
            next_token_logits = outputs.logits[output_idx].squeeze().cpu().numpy()
            
            # Store results
            hidden_states.append(final_hidden)
            logits.append(next_token_logits)

    # go back and get correct output for the initial test passes
    for i in range(n_test_passes):
        hidden_states[i] = hidden_states[i][output_idx]
        logits[i] = logits[i][output_idx]
    
    # Get valid answer tokens
    logits = np.array(logits)

    # top tokens
    k = len(config["valid_answers"]) * 4
    top_token_ids = np.sum(logits, axis=0).argsort()[::-1][:k]

    top_tokens = tokenizer.convert_ids_to_tokens(top_token_ids) 
    print("Top tokens:", top_tokens)

    # only keep top tokens
    logits = logits[:, top_token_ids]

    # Save results
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    np.save(output_dir / 'hidden_states.npy', np.array(hidden_states))
    np.save(output_dir / 'logits.npy', np.array(logits))

    # save the top tokens as a dict
    top_tokens_dict = {i: token_id_pair for i, token_id_pair in enumerate(zip(top_tokens, top_token_ids.tolist()))}
    with open(output_dir / 'top_tokens.json', 'w') as f:
        json.dump(top_tokens_dict, f)


def predict_with_mlm(config, model, tokenizer, text_data):
    """
    Predict using a masked language model and return the predictions, logits, and probabilities.
    """
    raise NotImplementedError("This code needs to be adapted")
    _set_seed(config.seed)

    # prepare data
    prompts = [
        RULE_TEMPLATE.format(text=item, rule=config.rule)
        for item in text_data
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(config.device)

    # Get prediction
    raw_results = []
    logits = []
    probs = []
    batch_size = config.batch_size
    for i in tqdm(range(0, len(inputs["input_ids"]), batch_size)):
        batch_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
        outputs = model(**batch_inputs)
        for j in range(len(batch_inputs["input_ids"])):
            mask_idx = (batch_inputs["input_ids"][j] == tokenizer.mask_token_id).nonzero()[0, 0]
            # store logit information
            logits_j = outputs.logits[j, mask_idx]
            probs_j = torch.softmax(logits_j, dim=0)

            top_idxs = torch.topk(probs_j, config.n_choices).indices.sort().values.tolist()
            logits.append(logits_j[top_idxs].detach().cpu().numpy())
            probs.append(probs_j[top_idxs].detach().cpu().numpy())

            # store prediction
            pred_id = logits_j.argmax()
            answer = tokenizer.decode(pred_id)
            raw_results.append(answer)

    preds = np.array(["true" in r.lower() for r in raw_results])
    logits = np.array(logits)
    probs = np.array(probs)

    return preds, logits, probs, raw_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)