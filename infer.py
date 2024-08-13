import json
import pathlib
import pickle
import transformers
import torch
import os
import copy
import wandb
import gc
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from config import *
from transformers import default_data_collator
from accelerate import DistributedDataParallelKwargs
import json
import pandas as pd
import os
from utils import *
from llama import Transformer, ModelArgs
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


def main(args, SEED):
    
    import torch
    import pickle
    from sklearn.model_selection import train_test_split

    # Create a tensor of numbers from 0 to 80
    data = torch.arange(276)

    # Shuffle the tensor
    shuffled_data = data[torch.randperm(data.size(0))]

    # Split into train, validation, and test sets (e.g., 60% train, 20% validation, 20% test)
    train_data, temp_data = train_test_split(shuffled_data, test_size=0.995)
    valid_data, test_data = train_test_split(temp_data, test_size=0.995)

    # Convert to tensors
    train_tensor = torch.tensor(train_data)
    valid_tensor = torch.tensor(valid_data)
    test_tensor = torch.tensor(test_data)

    # Save the tensors to a pickle file
    data_dict = {'train': train_tensor, 'valid': valid_tensor, 'test': test_tensor}

    with open('/home/sphaire/gnn_llm/trial/Graph-LLM/dataset/custom/split.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

    group = f"{args.dataset}"
    accelerator.init_trackers(project_name=f"{args.project}",
                              init_kwargs={"wandb":
                                               {"tags": [args.dataset, args.model_name],
                                                "group": group,
                                                "name": f"{args.dataset}_EXP{SEED}",
                                                "config": args}
                                           },
                              )

    seed_everything(seed=SEED)
    accelerator.print(args)


    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = 'right'

        dataset, split, edge_index = load_dataset[args.dataset](args.prediction_type)

        original_dataset = dataset.map(
            preprocess_original_dataset[args.dataset](tokenizer=tokenizer, max_length=original_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")

        clm_dataset_train = dataset.map(
            preprocess_train_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")


        clm_dataset_test = dataset.map(
            preprocess_test_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids', 'label', 'text_label']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")




    accelerator.wait_for_everyone()

    # Step 2: Build Node Classification Dataset
    train_dataset = clm_dataset_train.select(split['train'])
    val_dataset = clm_dataset_train.select(split['valid'])
    val_dataset_eval = clm_dataset_test.select(split['valid'])
    test_dataset = clm_dataset_test.select(split['test'])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                                               pin_memory=True, shuffle=True, collate_fn=default_data_collator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True,
                                             shuffle=False, collate_fn=default_data_collator)
    val_loader_eval = torch.utils.data.DataLoader(val_dataset_eval, batch_size=args.batch_size, drop_last=False,
                                                  pin_memory=True, shuffle=False, collate_fn=default_data_collator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False,
                                              pin_memory=True, shuffle=False, collate_fn=default_data_collator)


    with open(Path(f"{module_path}/{args.model_name}/") / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(w_lora=True,
                                      w_adapter=True,
                                      adapter_layer=8,
                                      adapter_dim=args.adapter_dim,
                                      adapter_len=args.adapter_len,
                                      lora_alpha=64,
                                      lora_r=128,
                                      num_hops=3,
                                      vocab_size = 10000,
                                      n_mp_layers=args.n_mp_layers,
                                      rrwp=args.rrwp,
                                      n_encoder_layers=args.n_encoder_layers,
                                      n_decoder_layers=args.n_decoder_layers,
                                      adapter_n_heads=args.adapter_n_heads,
                                      task_level=task_level[args.dataset],
                                      **params)


    model_args.vocab_size = tokenizer.vocab_size
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    base_model: Transformer = Transformer(params=model_args, edge_index=edge_index,
                                          input_ids=original_dataset['input_ids'],
                                          input_attention_mask=original_dataset['attention_mask'],
                                          )
    torch.set_default_tensor_type(torch.FloatTensor)

    accelerator.print(model_args)


    param_adapter, param_lora = base_model.set_trainable_params_new()


    lr_group = {
        'adapter': args.lr,
        'lora': args.lr,
    }

    wd_group = {
        'adapter': args.wd,
        'lora': args.wd,
    }

    accelerator.print(lr_group)
    accelerator.print(wd_group)

    optimizer = torch.optim.AdamW(
        [
            {'params': param_adapter, 'lr': lr_group['adapter'], 'weight_decay': wd_group['adapter']},
            {'params': param_lora, 'lr': lr_group['lora'], 'weight_decay': wd_group['lora']},
        ],
        betas=(0.9, 0.95))

    trainable_params, all_param = base_model.print_trainable_params()
    accelerator.print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    model, train_loader, val_loader, val_loader_eval, test_loader,optimizer = accelerator.prepare(base_model, train_loader,
                                                                                      val_loader, val_loader_eval,test_loader,
                                                                                      optimizer)
    ckpt = f"/home/sphaire/gnn_llm/trial/Graph-LLM/fine-tuned-gllama_next_10"
    accelerator.load_state(ckpt)
    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss, best_val_acc = float('inf'), 0.4
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)


    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    accelerator.wait_for_everyone()


    # Step 5. Evaluating


    samples_seen = 0
    eval_output = []
    node_ids = []
    model.eval()

    progress_bar_test = tqdm(range(len(test_loader)))

    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            kwargs = {}
            kwargs.update(
                {"node_ids": batch['node_ids'], "input_ids": batch['input_ids'],
                 "attention_mask": batch['attention_mask'], "max_new_tokens": 15})

            generated_tokens = accelerator.unwrap_model(model).generate(**kwargs)
            generated_tokens_gathered = accelerator.gather(generated_tokens).cpu().numpy()

            if accelerator.num_processes > 1:
                if step == len(test_loader) - 1:
                    generated_tokens_gathered = generated_tokens_gathered[: len(test_loader.dataset) - samples_seen]
                else:
                    samples_seen += len(generated_tokens_gathered)
            
            node_ids.extend(batch['node_ids'].cpu().numpy())
            print(len(node_ids))
            eval_output.append(generated_tokens_gathered)

        progress_bar_test.update(1)

    # Step 6. Post-processing & Evaluating
    if accelerator.is_local_main_process:
        eval_decode_output = []
        for batch_output in eval_output:
            eval_decode_output.extend(tokenizer.batch_decode(batch_output, skip_special_tokens=False))

        eval_pred = [item.split('<|begin_of_text|>You')[1] for item in eval_decode_output]
        # print(f' befor {eval_pred}')
        eval_pred = [item.split('\n\n###\n\n')[1] for item in eval_pred]
        eval_pred = [item.split('<|end_of_text|>')[0] for item in eval_pred]

        eval_label = test_loader.dataset['text_label']
        pred = [_ == f"{eval_label[i]}" for i, _ in enumerate(eval_pred)]
        print(pred)
        print(node_ids)
        print(f'eval pred is {len(eval_pred)} {eval_pred}')
        print(f'eval label is {len(eval_label)} {eval_label}')
        acc = sum(pred) / len(pred)

        accelerator.print(f'Test Acc {acc}')
        accelerator.log({'Test Acc': acc})


        file_path = f'/home/sphaire/gnn_llm/trial/Graph-LLM/dataset/custom_test/input_to_model_{args.prediction_type}.json'
        df = pd.read_json(file_path)
        df['turns'] = df['node_feat'].apply(lambda x: int((x.count('\n')+1)/2))
        df['label_frequency'] = df.groupby('label')['label'].transform('count')

        turns = []
        for i in node_ids:
            turns.append(df[df['node_ids'] == int(i)]['turns'].item())

        lf = []
        for i in node_ids:
            lf.append(df[df['node_ids'] == int(i)]['label_frequency'].item())

        item = {'node_id': node_ids,
                'turns': turns,
                'label_freq': lf,
                'next_node': pred}
        
        df = pd.DataFrame(item)
        df.to_csv(f'/home/sphaire/gnn_llm/trial/Graph-LLM/dataset/custom/eda_{args.prediction_type}.csv', index=False)

if __name__ == "__main__":

    args = parse_args_llama()
    for exp, SEED in enumerate(range(args.exp_num)):
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        transformers.logging.set_verbosity_error()
        accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs, init_kwargs],
                                  gradient_accumulation_steps=args.grad_steps)

        main(args, SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()