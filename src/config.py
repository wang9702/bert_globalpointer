import argparse


def set_args():
    parser = argparse.ArgumentParser('--bert_globalpointer实体识别')

    parser.add_argument('--data_dir', default='data/processed/', type=str, help='数据目录')
    parser.add_argument('--log_dir', default='log/', type=str, help='日志目录')
    parser.add_argument('--pre_model_dir', default='pretrained_model/chinese-bert-wwm', type=str, help='预训练模型的路径')
    parser.add_argument('--output_dir', default='models/', type=str, help='模型输出')
    parser.add_argument('--task_name', default='report_analyze', type=str, help='任务名称')
    parser.add_argument('--epochs', default=15, type=int, help='训练几轮')
    parser.add_argument('--train_batch_size', default=16, type=int, help='训练批次大小')
    parser.add_argument('--val_batch_size', default=16, type=int, help='验证批次大小')
    parser.add_argument('--max_len', default=256, type=int, help='序列最大长度')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累几次更新')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率大小')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='学习率warmup')
    parser.add_argument('--seed', default=2022, type=int, help='随机种子')
    parser.add_argument('--logging_step', default=50, type=int, help='日志打印步数')
    parser.add_argument('--device', default='cuda:1', type=str, help='模型输出')

    return parser.parse_args()


ent2id = {'DIAGNOSIS': 0, 'BP': 1, 'IS': 2, 'NW': 3}
# ent2id = {'dis': 0, 'sym': 1, 'pro': 2, 'equ': 3, 'dru': 4, 'ite': 5, 'bod': 6, 'dep': 7, 'mic': 8}
