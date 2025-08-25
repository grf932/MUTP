import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from four_model import ImagePatchEmbed, TextPositionEmbed
from four_model import FSRU_new
import higher
import pickle
import numpy as np

# 加载 w2v.pickle
with open('w2v.pickle', 'rb') as f:
    w2v = pickle.load(f)

# 自动判断并生成 vocab（word2idx）和 W（词向量表/或None）
first_value = list(w2v.values())[0]
if isinstance(first_value, int):
    vocab = w2v  # word2idx
    W = None     # 没有词向量矩阵
else:
    # 可能是list/np.ndarray（词向量），构造 word2idx 和 W
    vocab = {word: idx for idx, word in enumerate(w2v.keys())}
    W = np.stack([np.array(w2v[word]) for word in vocab.keys()])
    # W 的 shape: (vocab_size, embed_dim)
print(f"词表长度: {len(vocab)}，W 是否可用: {W is not None}, W.shape: {W.shape if W is not None else 'None'}")


# 或者你的主模型类名，如果不是FSRU_new请修改为你的主干模型名


# 假设你的主模型这两个类在这里

class ImgEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, d_model=128, dropout=0.):
        super().__init__()
        self.img_patch_embed = ImagePatchEmbed(img_size, patch_size, d_model)
        num_img_patches = self.img_patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, d_model))
        self.img_pos_drop = nn.Dropout(p=dropout)
        nn.init.trunc_normal_(self.img_pos_embed, std=.02)

    def forward(self, image):
        # image: [B, 3, H, W]
        x = self.img_patch_embed(image)
        x = x + self.img_pos_embed
        x = self.img_pos_drop(x)
        return x  # shape: (B, N, d_model)

class TxtEncoder(nn.Module):
    def __init__(self, vocab_size, d_text, seq_len, d_model=128, W=None, dropout=0.):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, d_text)
        if W is not None:
            self.text_embed.weight = nn.Parameter(torch.from_numpy(W))
        self.text_encoder = nn.Sequential(
            nn.Linear(d_text, d_model),
            nn.LayerNorm(d_model),
            TextPositionEmbed(seq_len, d_model, dropout)
        )

    def forward(self, text):
        # text: [B, seq_len] (LongTensor)
        x = text.long()
        x = self.text_embed(x)
        x = self.text_encoder(x)
        return x  # shape: (B, seq_len, d_model)

class DataRaterModel(nn.Module):
    def __init__(self, img_size=256, patch_size=16, d_model=128, dropout=0.1,
                 vocab_size=1734, d_text=128, seq_len=184, W=None, transformer_dim=128, num_layers=2):
        super().__init__()
        # 1. 用主模型同款编码器
        self.img_encoder = ImgEncoder(img_size, patch_size, d_model, dropout)
        self.txt_encoder = TxtEncoder(vocab_size, d_text, seq_len, d_model, W, dropout)

        # 2. 融合交互 Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=8, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 输出评分头
        self.head = nn.Sequential(
            nn.Linear(transformer_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 4. 图片预处理
        from torchvision import transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def forward(self, img_pil, text_tensor):
        """
        img_pil: PIL.Image 或 batch list
        text_tensor: torch.LongTensor (B, seq_len)
        """
        # --- 图像编码 ---
        if isinstance(img_pil, list):
            img_tensor = torch.stack([self.img_transform(im) for im in img_pil])
        else:
            img_tensor = self.img_transform(img_pil).unsqueeze(0) if len(img_pil.shape) == 3 else img_pil
        img_feat = self.img_encoder(img_tensor)  # (B, N, d_model)
        img_feat_pooled = img_feat.mean(dim=1)   # (B, d_model)

        # --- 文本编码 ---
        txt_feat = self.txt_encoder(text_tensor)  # (B, seq_len, d_model)
        txt_feat_pooled = txt_feat.mean(dim=1)    # (B, d_model)

        # --- 融合输入到Transformer ---
        multi_feat = torch.stack([img_feat_pooled, txt_feat_pooled], dim=1)  # (B, 2, d_model)
        trans_input = multi_feat.permute(1, 0, 2)                            # (seq_len=2, B, d_model)
        out = self.transformer(trans_input)                                  # (2, B, d_model)
        pooled = out.mean(dim=0)                                             # (B, d_model)

        # --- 评分 ---
        score = self.head(pooled).squeeze(-1)                                # (B,)
        return score


class InnerTaskModel(nn.Module):
    def __init__(self, W, vocab_size, d_text, seq_len, img_size, patch_size, d_model, num_filter, num_class, num_layer, **kwargs):
        """
        初始化时各参数与你的主模型一致
        - W: 词向量矩阵
        - vocab_size, d_text, seq_len: 文本相关参数
        - img_size, patch_size, d_model: 图像相关参数
        - num_filter: 卷积/特征过滤器数量
        - num_class: 分类类别数
        - num_layer: 层数
        - kwargs: 其他主模型初始化参数
        """
        super().__init__()
        self.model = FSRU_new(
            W, vocab_size, d_text, seq_len, img_size, patch_size, d_model,
            num_filter, num_class, num_layer, **kwargs
        )

    def forward(self, text, image):
        """
        text: torch.LongTensor [B, seq_len]
        image: torch.FloatTensor [B, 3, H, W]
        返回主模型的输出（如 logits）
        """
        return self.model(text, image)


class MetaTrainer:
    """
    用higher实现的DataRater元学习训练器
    """
    def __init__(
        self, datarater, inner_model, train_loader, val_loader,
        device='cuda', inner_steps=1, inner_lr=1e-3, meta_lr=1e-4
    ):
        self.datarater = datarater
        self.inner_model = inner_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_optimizer = optim.Adam(self.datarater.parameters(), lr=meta_lr)

    def run_meta_epoch(self, criterion):
        self.datarater.train()
        for batch in self.train_loader:
            img, text, label = batch
            img, text, label = img.to(self.device), text.to(self.device), label.to(self.device)

            # === 1. 内层优化器和higher上下文 ===
            inner_optimizer = optim.SGD(self.inner_model.parameters(), lr=self.inner_lr)
            with higher.innerloop_ctx(self.inner_model, inner_optimizer, copy_initial_weights=True) as (fmodel, diffopt):

                # === 2. DataRater对当前batch评分 ===
                scores = self.datarater(img, text)         # [B]
                weights = torch.sigmoid(scores)
                weights = weights / (weights.sum() + 1e-8)

                # === 3. 内循环（T步）===
                for _ in range(self.inner_steps):
                    logits = fmodel(text, img)             # [B, C]
                    # 确保logits为Tensor
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss_vec = criterion(logits, label)    # [B]
                    weighted_loss = (loss_vec * weights).sum()
                    diffopt.step(weighted_loss)

                # === 4. 验证集上算外层损失 ===
                val_img, val_text, val_label = next(iter(self.val_loader))
                val_img, val_text, val_label = val_img.to(self.device), val_text.to(self.device), val_label.to(self.device)
                val_logits = fmodel(val_text, val_img)
                # 确保val_logits为Tensor
                if isinstance(val_logits, tuple):
                    val_logits = val_logits[0]
                val_loss = criterion(val_logits, val_label)
                val_loss = val_loss.mean()

                # === 5. 更新DataRater ===
                self.meta_optimizer.zero_grad()
                val_loss.backward()
                self.meta_optimizer.step()

        print("One meta-epoch (with higher) finished.")

    def train(self, criterion, meta_epochs=10):
        for epoch in range(meta_epochs):
            self.run_meta_epoch(criterion)
            print(f"Meta epoch {epoch+1}/{meta_epochs} finished.")


class MultimodalDataset(Dataset):
    """
    适配FSRU输入格式的多模态Dataset
    """
    def __init__(self, csv_path, image_root, vocab, seq_len=184, img_size=256, img_transform=None):
        """
        csv_path: CSV路径，包含image_url, tweet, label等字段
        image_root: 图片根目录
        vocab: 字符串到token的字典（word2idx）
        seq_len: 文本最大长度
        img_size: 图像尺寸
        img_transform: 图像预处理transform
        """
        self.samples = []
        self.vocab = vocab
        self.seq_len = seq_len
        self.img_size = img_size
        self.img_transform = img_transform
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "img_path": os.path.join(image_root, row['image_url']),
                    "image_url": row['image_url'],
                    "tweet": row['tweet'],
                    "label": int(row['label']) if 'label' in row else 0
                })

    def __len__(self):
        return len(self.samples)

    def tokenize(self, tweet):
        # 假设vocab为 word2idx 字典
        tokens = [self.vocab.get(w, 1) for w in tweet.lower().split()]  # 1=UNK
        tokens = tokens[:self.seq_len]
        tokens += [0] * (self.seq_len - len(tokens))
        return torch.LongTensor(tokens)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        if self.img_transform:
            img = self.img_transform(img)
        text = self.tokenize(sample["tweet"])
        label = sample["label"]
        return img, text, label

class DatasetManager:
    """
    负责数据加载、分割、dataloader创建
    """
    def __init__(self, csv_path, image_root, vocab, batch_size=32, seq_len=184, img_size=256, img_transform=None, val_ratio=0.1, shuffle=True):
        dataset = MultimodalDataset(csv_path, image_root, vocab, seq_len, img_size, img_transform)
        n = len(dataset)
        n_val = int(n * val_ratio)
        indices = torch.randperm(n) if shuffle else torch.arange(n)
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        self.train_set = torch.utils.data.Subset(dataset, train_idx)
        self.val_set = torch.utils.data.Subset(dataset, val_idx)
        self.batch_size = batch_size

    def get_train_loader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)



class FilterEngine:
    """
    用训练好的DataRaterModel筛选优质样本。
    支持 Top-K、Top-pct、阈值三种筛选方式，并可导出csv或返回样本索引。
    """
    def __init__(self, datarater_model, device='cuda'):
        self.datarater = datarater_model.to(device)
        self.device = device

    def score_dataset(self, dataset, batch_size=32):
        """
        给dataset里的每个样本打分。
        返回：所有分数的list，长度=len(dataset)
        """
        self.datarater.eval()
        all_scores = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Scoring dataset"):
                imgs, texts, _ = batch  # 忽略label
                imgs = imgs.to(self.device)
                texts = texts.to(self.device)
                scores = self.datarater(imgs, texts)  # [B]
                all_scores.append(scores.cpu())
        all_scores = torch.cat(all_scores, dim=0)
        return all_scores.numpy()  # (N,)

    def filter_top_k(self, dataset, scores, k):
        """
        保留分数最高的k个样本。
        """
        idx = scores.argsort()[::-1][:k]  # 降序取前k
        return [dataset.samples[i] for i in idx]

    def filter_top_pct(self, dataset, scores, pct=0.2):
        """
        保留分数最高的前pct比例（如pct=0.2为前20%）。
        """
        n = int(len(scores) * pct)
        return self.filter_top_k(dataset, scores, n)

    def filter_by_threshold(self, dataset, scores, threshold):
        """
        只保留分数大于threshold的样本。
        """
        idx = [i for i, s in enumerate(scores) if s > threshold]
        return [dataset.samples[i] for i in idx]

    def filter_top_pct_per_label(self, dataset, scores, pct=0.2, num_classes=4):
        """
        每个类别分别筛选前pct比例高分样本。
        返回合并后的样本列表。
        """
        # 首先收集每个类别的样本索引
        label_to_indices = {label: [] for label in range(num_classes)}
        for i, sample in enumerate(dataset.samples):
            label = sample['label']
            if label in label_to_indices:
                label_to_indices[label].append(i)
        selected_indices = []
        for label, indices in label_to_indices.items():
            if not indices:
                continue
            label_scores = [scores[i] for i in indices]
            n_label = max(1, int(len(indices) * pct))
            # 对当前类别的indices按分数排序，取前n_label
            sorted_indices = [x for _, x in sorted(zip(label_scores, indices), reverse=True)]
            selected_indices.extend(sorted_indices[:n_label])
        # 返回所有选中样本
        return [dataset.samples[i] for i in selected_indices]

    def export_csv(self, samples, out_csv_path):
        """
        保存筛选后的样本为新的csv（image_url, tweet, label）。
        """
        with open(out_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_url', 'tweet', 'label'])
            writer.writeheader()
            for sample in samples:
                writer.writerow({
                    'image_url': sample['image_url'],
                    'tweet': sample['tweet'],
                    'label': sample['label']
                })

    def filter_and_export(
        self, dataset, method='top_pct', value=0.2, out_csv='filtered_data.csv', batch_size=32
    ):
        """
        一键筛选并导出。
        method: 'top_k', 'top_pct', 'threshold', 'top_pct_per_label'
        value: 依赖于method的参数
        """
        scores = self.score_dataset(dataset, batch_size=batch_size)
        if method == 'top_k':
            filtered = self.filter_top_k(dataset, scores, int(value))
        elif method == 'top_pct':
            filtered = self.filter_top_pct(dataset, scores, float(value))
        elif method == 'threshold':
            filtered = self.filter_by_threshold(dataset, scores, float(value))
        elif method == 'top_pct_per_label':
            filtered = self.filter_top_pct_per_label(dataset, scores, float(value))
        else:
            raise ValueError("Unknown filter method!")
        self.export_csv(filtered, out_csv)
        print(f"筛选后样本数：{len(filtered)}，已保存到 {out_csv}")
        return filtered


import argparse
from torchvision import transforms

def main():
    # ========= 参数设置 =========
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='accident_captions.csv')
    parser.add_argument('--image_root', type=str, default='path_to_images')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--meta_epochs', type=int, default=10)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--filter_method', type=str, default='top_pct')  # or 'top_k' or 'threshold'
    parser.add_argument('--filter_value', type=float, default=0.2)
    parser.add_argument('--filtered_csv', type=str, default='filtered_data.csv')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # ========= 数据准备 =========
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    dm = DatasetManager(
        csv_path=args.csv_path,
        image_root=args.image_root,
        vocab=vocab,
        batch_size=args.batch_size,
        seq_len=184,
        img_size=256,
        img_transform=img_transform,
        val_ratio=args.val_ratio,
        shuffle=True
    )
    train_loader = dm.get_train_loader()
    val_loader = dm.get_val_loader()
    # 用完整数据集评分和筛选
    full_dataset = MultimodalDataset(args.csv_path, args.image_root, vocab, seq_len=184, img_size=256, img_transform=img_transform)

    # ========= 模型初始化 =========
    # 根据W决定embedding维度
    embed_dim = W.shape[1] if W is not None else 128
    datarater = DataRaterModel(
        img_size=256, patch_size=16, d_model=128, dropout=0.1,
        vocab_size=len(vocab), d_text=embed_dim, seq_len=184, W=W,  # 如果有W，传W
        transformer_dim=128, num_layers=2
    ).to(args.device)

    inner_model = InnerTaskModel(
        W=W, vocab_size=len(vocab), d_text=embed_dim, seq_len=184,
        img_size=256, patch_size=16, d_model=128,
        num_filter=2, num_class=4, num_layer=4
    ).to(args.device)

    # ========= 损失函数 =========
    criterion = nn.CrossEntropyLoss(reduction='none')

    # ========= meta-training =========
    meta_trainer = MetaTrainer(
        datarater, inner_model, train_loader, val_loader,
        device=args.device, inner_steps=1, inner_lr=1e-3, meta_lr=1e-4
    )
    print("开始meta-training...")
    meta_trainer.train(criterion, meta_epochs=args.meta_epochs)
    print("meta-training完成。")

    # ========= 用FilterEngine筛选优质样本 =========
    filter_engine = FilterEngine(datarater, device=args.device)
    filter_engine.filter_and_export(
        dataset=full_dataset,
        method=args.filter_method,
        value=args.filter_value,
        out_csv=args.filtered_csv,
        batch_size=args.batch_size
    )
    print(f"数据筛选完毕，新csv已保存为 {args.filtered_csv}")

if __name__ == '__main__':
    main()