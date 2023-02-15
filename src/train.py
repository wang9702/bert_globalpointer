# -*- coding: utf-8 -*-
import os
import torch
from src.features import EntDataset
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from src.model import EffiGlobalPointer
from src.evaluate import MetricsCalculator
from tqdm import tqdm
from src.config import set_args, ent2id
from src.utils import set_logger, set_seed, EarlyStopping
from torch.optim import AdamW


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss

def train_and_evaluate(args, model, optimizer, scheduler, ner_loader_train, ner_loader_evl):
    metrics = MetricsCalculator()
    # max_f, max_recall = 0.0, 0.0
    # patience_num = 0
    early_stopping = EarlyStopping(args.output_dir, 2, logger=logger)
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_f1 = 0., 0.
        for idx, batch in enumerate(ner_loader_train):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(args.device), attention_mask.to(args.device), segment_ids.to(args.device), labels.to(args.device)
            logits = model(input_ids, attention_mask, segment_ids)
            loss = loss_fun(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            scheduler.step()
            sample_f1 = metrics.get_sample_f1(logits, labels)
            total_loss+=loss.item()
            total_f1 += sample_f1.item()

            avg_loss = total_loss / (idx + 1)
            avg_f1 = total_f1 / (idx + 1)
            if idx % args.logging_step == 0:
                logger.info('epoch:%d\t step:%d/%d\t trian_loss:%f\t train_f1:%f'%(epoch, idx, len(ner_loader_train), avg_loss, avg_f1))

        with torch.no_grad():
            model.eval()
            pred_logits = []
            true_labels = []
            for batch in tqdm(ner_loader_evl, desc="Valing"):
                raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
                input_ids, attention_mask, segment_ids, labels = input_ids.to(args.device), attention_mask.to(
                    args.device), segment_ids.to(args.device), labels.to(args.device)
                logits = model(input_ids, attention_mask, segment_ids)
                pred_logits.append(logits)
                true_labels.append(labels)
            f1, p, r, report = metrics.get_evaluate_fpr(pred_logits, true_labels)
            logger.info("Epoch:%d\tF1:%f\tPrecision:%f\tRecall:%f\t"%(epoch, round(f1, 5), round(p, 5), round(r, 5)))
            logger.info(f'Classification Report: \n{report}')
            # improve_f1 = f1 - max_f
            # model_to_save = model.module if hasattr(model, 'module') else model
            # if improve_f1 > 0:
            #     torch.save(model_to_save.state_dict(), f'{args.output_dir}/model.pth')
            #     logger.info("Best val f1: {:05.4f}".format(f1))
            #     max_f = f1
            #     patience_num = 0
            # else:
            #     patience_num += 1
            #     if patience_num == 2:
            #         logger.info("Best val f1: {:05.4f}".format(max_f))
            #         break
            early_stopping(f1, model)
            if early_stopping.early_stop:
                logger.info("Early Stopping")
                break 


def main(args):
    # 模型保存路径
    os.makedirs(os.path.join(args.output_dir, args.task_name), exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.task_name)
    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.pre_model_dir, do_lower_case=True)
    # 训练集和验证集
    train_path = os.path.join(args.data_dir, args.task_name, 'train.json')
    eval_path = os.path.join(args.data_dir, args.task_name, 'dev.json')
    train_dataset = EntDataset(train_path, args, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate, shuffle=True)
    eval_dataset = EntDataset(eval_path, args, tokenizer=tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=args.val_batch_size, collate_fn=eval_dataset.collate, shuffle=False)
    # GP MODEL
    encoder = BertModel.from_pretrained(args.pre_model_dir)
    model = EffiGlobalPointer(encoder, len(ent2id), 64).to(args.device)
    # optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': args.learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    num_train_optimization_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                int(num_train_optimization_steps * args.warmup_proportion),
                                                num_train_optimization_steps)
    train_and_evaluate(args, model, optimizer, scheduler, train_loader, eval_loader)


if __name__=='__main__':

    args = set_args()
    set_seed(args.seed)
    os.makedirs(os.path.join(args.log_dir, args.task_name), exist_ok=True)
    log_dir = os.path.join(args.log_dir, args.task_name)
    logger = set_logger(log_path=os.path.join(log_dir, 'train.log'))
    main(args)



    
    
    


