import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from Modular_Transformer import make_model
from tqdm import tqdm

# =======================
# 1. 数据集类（wmt16 版本）
# =======================
class WMT16Dataset(Dataset):
    def __init__(self, split="train", src_lang="de", tgt_lang="en", max_len=50, max_samples=500, data_dir="./data"):
        dataset = load_dataset("wmt16", "de-en", split=split, cache_dir=data_dir)
        if max_samples:
            dataset = dataset.select(range(max_samples))

        # ⚠️ 注意这里访问 translation 字段
        self.src_sentences = [ex['translation'][src_lang] for ex in dataset]
        self.tgt_sentences = [ex['translation'][tgt_lang] for ex in dataset]
        self.max_len = max_len

        # 构建词表
        # 词表大小 = 所有训练句子中不重复的 token 数 + 特殊 token。
        # 原本的每个token就是在词表中的一个数字，通过embedding后，变为d_model长度的向量了
        self.src_vocab = self.build_vocab(self.src_sentences)
        self.tgt_vocab = self.build_vocab(self.tgt_sentences)

    @staticmethod
    def build_vocab(sentences):
        vocab = {"<pad>":0, "<sos>":1, "<eos>":2}
        idx = 3
        for s in sentences:
            for w in s.strip().split():
                if w not in vocab:
                    vocab[w] = idx
                    idx += 1
        return vocab

    def encode_sentence(self, sentence, vocab):
        tokens = ["<sos>"] + sentence.strip().split() + ["<eos>"]
        ids = [vocab.get(t, 0) for t in tokens]
        if len(ids) < self.max_len:
            ids += [vocab["<pad>"]] * (self.max_len - len(ids))
        return ids[:self.max_len]

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_ids = torch.tensor(self.encode_sentence(self.src_sentences[idx], self.src_vocab))
        tgt_ids = torch.tensor(self.encode_sentence(self.tgt_sentences[idx], self.tgt_vocab))
        return src_ids, tgt_ids

# =======================
# 2. Collate 函数
# =======================
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return src_batch, tgt_batch

# =======================
# 3. Mask 函数
# =======================
def create_masks(src, tgt, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(1)
    tgt_len = tgt.size(1)
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=src.device)).bool()
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1)
    tgt_mask = tgt_mask & tgt_padding_mask
    return src_mask, tgt_mask

# =======================
# 4. 训练函数
# =======================
def train(model, dataset, epochs=5, lr=1e-3, batch_size=32, save_dir="model", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_vocab["<pad>"])

    start_epoch = 0
    save_path = os.path.join(save_dir, "transformer_wmt16.pth")

    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Loaded model from epoch {checkpoint['epoch']+1}, loss={checkpoint['loss']:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        # 使用 tqdm 包裹 DataLoader，显示进度
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for src, tgt in loop:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = create_masks(src, tgt[:, :-1], dataset.src_vocab["<pad>"])

            optimizer.zero_grad()
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)
            logits = model.generator(output)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 更新进度条描述显示当前 loss
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss = {avg_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, save_path)
        print(f"💾 Model saved at {save_path}")

# =======================
# 5. 推理函数
# =======================
def translate(model, sentence, dataset, max_len=50, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        src_ids = torch.tensor(dataset.encode_sentence(sentence, dataset.src_vocab)).unsqueeze(0).to(device)
        src_mask = (src_ids != dataset.src_vocab["<pad>"]).unsqueeze(1).to(device)
        tgt_ids = torch.tensor([[dataset.tgt_vocab["<sos>"]]], device=device)

        for _ in range(max_len):
            tgt_mask = torch.tril(torch.ones(tgt_ids.size(1), tgt_ids.size(1), device=device)).bool().unsqueeze(0)
            out = model(src_ids, tgt_ids, src_mask, tgt_mask)
            logits = model.generator(out)
            next_word = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            tgt_ids = torch.cat([tgt_ids, next_word], dim=1)
            if next_word.item() == dataset.tgt_vocab["<eos>"]:
                break

        id_to_word = {i: w for w, i in dataset.tgt_vocab.items()}
        words = [id_to_word[i.item()] for i in tgt_ids[0][1:]]
        if "<eos>" in words:
            words = words[:words.index("<eos>")]
        return " ".join(words)

# =======================
# 6. 主程序
# =======================
if __name__ == "__main__":
    dataset = WMT16Dataset(split="train", src_lang="de", tgt_lang="en", max_len=500, max_samples=10000, data_dir="./data")
    '''
    参数说明：
    split="train"     -> 使用训练集（train），可选 "validation" 或 "test"    
        train       454885条
        validation  2169条
        test        2999条
    src_lang="de"     -> 源语言为德语，模型输入为德语句子
    tgt_lang="en"     -> 目标语言为英语，模型输出为英语句子
    max_len=50        -> 每个句子最大长度，超过截断，不足补 <pad>
    max_samples=5000  -> 最大取样数量为 5000 条，用于快速调试
    data_dir="./data" -> 数据下载/缓存到项目目录下的 ./data 文件夹
    '''

    model = make_model(len(dataset.src_vocab), len(dataset.tgt_vocab), N=2, d_model=512, d_ff=1024, h=4)
    '''
    参数说明：
    len(dataset.src_vocab) -> 源语言词表大小（德语），用于嵌入层输入维度
    len(dataset.tgt_vocab) -> 目标语言词表大小（英语），用于输出层生成概率
    N=2   -> Encoder 和 Decoder 堆叠的层数，每个堆叠包含多头注意力和前馈网络
    d_model=128 -> Transformer 中隐藏状态向量维度，也是嵌入向量维度
    d_ff=256    -> 前馈全连接层的维度（Feed-Forward Network）
    h=4        -> 多头注意力机制中的头数（head 数），每个头独立计算注意力
    '''

    train(model, dataset, epochs=30, batch_size=32, save_dir="model")


    while True:
        sentence = input("Enter German sentence (or 'quit' to exit): ")
        if sentence.lower() == "quit":
            break
        translation = translate(model, sentence, dataset)
        print("Translation:", translation)
    '''
    data = [
        ("i like cats", "ich mag katzen"),
        ("you love dogs", "du liebst hunde"),
        ("he eats food", "er isst essen"),
        ("we play games", "wir spielen spiele"),
        ("they read books", "sie lesen bücher"),
    ]
    '''