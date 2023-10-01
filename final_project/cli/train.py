# +
import datasets
import transformers
from datasets import load_dataset
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from transformers import AdamW, MT5ForConditionalGeneration, MT5Tokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import wandb
import logging
import sacrebleu
import os
from packaging import version

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_len",default=20, type=int)
parser.add_argument("--batch_size",default=8, type=int)
parser.add_argument("--num_train_epochs",default=1, type=int)
parser.add_argument("--learning_rate",default=5e-5, type=float)
parser.add_argument("--dataset", default='code_x_glue_tt_text_to_text', type=str)
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--source_lang", default='da', type=str)
parser.add_argument("--target_lang", default='en', type=str)
parser.add_argument("--dataset_config_name", default='da_en', type=str)
parser.add_argument("--output_dir", default='da_en_output_dir', type=str)
parser.add_argument("--debug", default=False, action="store_true",
        help="Whether to use a small subset of the dataset for debugging.")

# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_warning()

args = parser.parse_args()

def main():
    logger.info(f"Starting script with arguments: {args}")
    
    # Initialize wandb as soon as possible to log all stdout to the cloud
    wandb.init(project="final_project", config=args)
    
    model_repo = 'google/mt5-small'
    output_dir = args.output_dir
    max_seq_len = args.max_seq_len
    device = torch.device(args.device)
    
    tokenizer = MT5Tokenizer.from_pretrained(model_repo)
    model = MT5ForConditionalGeneration.from_pretrained(model_repo)
    model = model.to(device)
    
    datasets = load_dataset(args.dataset, args.dataset_config_name)
    
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    
    LANG_TOKEN_MAPPING = {
        args.source_lang: '<'+args.source_lang+'>',
        args.target_lang: '<'+args.target_lang+'>'
    }
    
    special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    # Constants
    n_epochs = args.num_train_epochs
    batch_size = args.batch_size
    print_freq = 50
    checkpoint_freq = 1000
    lr = args.learning_rate
    n_batches = int(np.ceil(len(train_dataset) / batch_size))
    total_steps = n_epochs * n_batches
    n_warmup_steps = int(total_steps * 0.01)
    
    losses = []
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, n_warmup_steps, total_steps)
    
    def encode_input_str(text, target_lang, tokenizer, seq_len,
                         lang_token_map=LANG_TOKEN_MAPPING):
        target_lang_token = lang_token_map[target_lang]

      # Tokenize and add special tokens
        input_ids = tokenizer.encode(
              text = str(target_lang_token) + str(text),
              return_tensors = 'pt',
              padding = 'max_length',
              truncation = True,
              max_length = seq_len)

        return input_ids[0]
    
    def encode_target_str(text, tokenizer, seq_len,
                          lang_token_map=LANG_TOKEN_MAPPING):
        token_ids = tokenizer.encode(
              text = text,
              return_tensors = 'pt',
              padding = 'max_length',
              truncation = True,
              max_length = seq_len)

        return token_ids[0]
    
    
    def format_translation_data(translations, lang_token_map,
                                tokenizer, seq_len=128):
        # Choose languages for in i/o
        input_lang, target_lang = [args.source_lang, args.target_lang]

        # Get the translations for the batch
        input_text = translations["source"]
        target_text = translations["target"]

        if input_text is None or target_text is None:
            return None

        input_token_ids = encode_input_str(
        input_text, target_lang, tokenizer, seq_len, lang_token_map)

        target_token_ids = encode_target_str(
        target_text, tokenizer, seq_len, lang_token_map)

        return input_token_ids, target_token_ids
    
    
    def transform_batch(batch, lang_token_map, tokenizer):
        inputs = []
        targets = []

        formatted_data = format_translation_data(batch, lang_token_map, tokenizer, max_seq_len)

        input_ids, target_ids = formatted_data
        inputs.append(input_ids.unsqueeze(0))
        targets.append(target_ids.unsqueeze(0))

        batch_input_ids = torch.cat(inputs).to(device)
        batch_target_ids = torch.cat(targets).to(device)

        return batch_input_ids, batch_target_ids
    
    def get_data_generator(dataset, lang_token_map, tokenizer, batch_size):
        dataset = dataset.shuffle()
        for i in range(0, len(dataset), batch_size):
            raw_batch = dataset[i:i+batch_size]
            yield transform_batch(raw_batch, lang_token_map, tokenizer)
            
    def eval_model(model, gdataset, max_iters=8):
        test_generator = get_data_generator(gdataset, LANG_TOKEN_MAPPING,
                                          tokenizer, batch_size)
        eval_losses = []
        for i, (input_batch, label_batch) in enumerate(test_generator):
            if i >= max_iters:
                break

            # passes and weights update
            with torch.set_grad_enabled(True):
                model_out = model.forward(
                    input_ids = input_batch.to(device),
                    labels = label_batch.to(device))
                eval_losses.append(model_out.loss.item())

        return np.mean(eval_losses)
    
    
    def eval_bleu(model, test_dataset):
        english_truth = test_dataset["target"]
        to_english = test_dataset["source"]

        english_preds=[]
        for i in to_english:
            # passes and weights update
            with torch.set_grad_enabled(True):
                input_ids = encode_target_str(
                    text = i,
                    tokenizer = tokenizer,
                    seq_len = model.config.max_length)
                input_ids = input_ids.unsqueeze(0).cuda()
                output_tokens = model.generate(input_ids, num_beams=10, num_return_sequences=1, length_penalty = 1, no_repeat_ngram_size=2)
                for token_set in output_tokens:
                    english_preds.append(tokenizer.decode(token_set, skip_special_tokens=True))

        bleu = sacrebleu.corpus_bleu(english_preds, english_truth)

        return bleu.score
    
    
    for epoch_idx in range(n_epochs):
        # Randomize data order
        data_generator = get_data_generator(train_dataset, LANG_TOKEN_MAPPING,
                                          tokenizer, batch_size)

        for batch_idx, (input_batch, label_batch) in tqdm(enumerate(data_generator), total=n_batches):

            # passes and weights update
            with torch.set_grad_enabled(True):

                # Forward pass
                model_out = model.forward(input_ids = input_batch.to(device), labels = label_batch.to(device))

                # Calculate loss and update weights
                loss = model_out.loss
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Print training update info
                if (batch_idx + 1) % print_freq == 0:
                    avg_loss = np.mean(losses[-print_freq:])
                    print('Epoch: {} | Step: {} | Avg. loss: {:.3f} | lr: {}'.format(
                        epoch_idx+1, batch_idx+1, avg_loss, scheduler.get_last_lr()[0]))
                    wandb.log(
                        {
                            "train_loss": avg_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch_idx,
                        },
                    )
                    model.save_pretrained(output_dir)

                if (batch_idx + 1) % checkpoint_freq == 0:
                    test_loss = eval_model(model, test_dataset)
                    print('Saving model with test loss of {:.3f}'.format(test_loss))
                    wandb.log(
                        {
                            "train_loss": test_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch_idx,
                        },
                    )
                    model.save_pretrained(output_dir)
                    
        bleu_score = eval_bleu(model, test_dataset)
        wandb.log({"eval/bleu": bleu_score})
        model.save_pretrained(output_dir)
    
    logger.info("Saving final model checkpoint to %s", output_dir)
    model.save_pretrained(output_dir)

    logger.info("Uploading tokenizer, model and config to wandb")
    wandb.save(os.path.join(output_dir, "*"))

    logger.info(f"Script finished succesfully, model saved in {output_dir}")
    
if __name__ == "__main__":
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")

    main()
