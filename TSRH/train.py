import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from data_factory import DataFactory
from model.model_factory import ModelFactory
from metrics import t2v_v2t_metrics, calculate_map_t2v, calculate_map_v2t
from loss import LossFactory
from trainer import Trainer
from optimization import AdamW, get_cosine_schedule_with_warmup
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    config = AllConfig()
    for k, v in vars(config).items():
        print(f"{k}: {v}")
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.huggingface:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    else:
        from modules.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()
    from time import time  
    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    valid_data_loader  = DataFactory.get_data_loader(config, split_type='test')

    model = ModelFactory.get_model(config)
    if config.metric == 't2v':
        metrics = t2v_v2t_metrics
    elif config.metric == 'v2t':
        metrics = t2v_v2t_metrics
    elif config.metric == 'map_t2v':
        metrics = calculate_map_t2v
    elif config.metric == 'map_v2t':
        metrics = calculate_map_v2t
    else:
        raise NotImplemented   
    params_optimizer = list(model.named_parameters())

    clip_params = [p for n, p in params_optimizer if "clip." in n]  
    noclip_params = [p for n, p in params_optimizer if "clip." not in n] 

    optimizer_grouped_params = [
        {'params': clip_params, 'lr': config.clip_lr},
        {'params': noclip_params, 'lr': config.noclip_lr}
    ]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    
    loss = LossFactory.get_loss(config)
    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=writer,
                      tokenizer=tokenizer)

    trainer.train()

if __name__ == '__main__':
    main()
