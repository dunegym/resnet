# finetune_qwen_0_6b_final.py
"""
Qwen-0.6B 文本分类微调 | 终极修复版
✅ 解决：LoRA 无梯度、FP16 缩放错误、pad_token 问题
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =======================
# 🛡️ 强制屏蔽 bitsandbytes（避免 CUDA 错误）
# =======================
import sys
from types import ModuleType

def create_fake_bitsandbytes():
    if "bitsandbytes" in sys.modules:
        return
    bnb = ModuleType("bitsandbytes")
    bnb.__spec__ = type('spec', (), {'loader': None})()
    bnb.__file__ = None
    bnb.nn = ModuleType("bitsandbytes.nn")
    bnb.nn.modules = ModuleType("bitsandbytes.nn.modules")
    sys.modules["bitsandbytes"] = bnb
    sys.modules["bitsandbytes.nn"] = bnb.nn
    sys.modules["bitsandbytes.nn.modules"] = bnb.nn.modules

create_fake_bitsandbytes()

# =======================
# ✅ 导入依赖
# =======================
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# =======================
# 1. 路径配置
# =======================
model_path = "../Qwen3-0.6B"
data_path = "train_processed.parquet"
output_dir = "./qwen-0.6b-prompt-sentiment-lora"

# =======================
# 2. 加载 tokenizer
# =======================
print("🔄 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# ✅ 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"  # 推荐

print(f"✅ pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

# =======================
# 3. 加载模型（FP16 + GPU）
# =======================
print("🔄 加载模型...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,  # 二分类：positive vs negative/neutral
    torch_dtype=torch.float16,      # 显存优化
    device_map="auto",              # 自动分配
    trust_remote_code=True
)

# ✅ 关键：设置 model.config.pad_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id
print(f"✅ model.config.pad_token_id: {model.config.pad_token_id}")

# =======================
# 4. LoRA 配置 + 强制启用梯度
# =======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",        # 自动匹配所有线性层
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["score"],          # 保留分类头
    task_type="SEQ_CLS"
)

print("🔧 添加 LoRA 适配器...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ✅ 强制启用 score 层梯度（关键！）
print("✅ 强制启用 score 层梯度")
model.score.requires_grad_(True)

# ✅ 确保所有 LoRA 参数可训练
print("🔍 确保 LoRA 参数可训练...")
for name, param in model.named_parameters():
    if "lora_" in name or "score" in name:
        param.requires_grad = True

# 再次检查
print("\n✅ 最终可训练参数：")
model.print_trainable_parameters()  # 应显示 > 0

# =======================
# 5. 加载数据集
# =======================
print("📊 加载数据集...")
dataset = load_dataset("parquet", data_files=data_path, split="train")
print(f"✅ 数据集大小: {len(dataset):,}")

# 标签映射
def map_sentiment(label):
    if isinstance(label, str):
        return 1 if label.strip().lower() == "positive" else 0
    return 0

# 分词
def tokenize_function(examples):
    texts = examples["prompt"]
    labels = [map_sentiment(lbl) for lbl in examples["prompt_sentiment"]]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None,
        return_attention_mask=True
    )
    encodings["labels"] = labels
    return encodings

# 处理
remove_columns = [
    "prompt", "response_a", "response_b", "prompt_sentiment",
    "id", "model_a", "model_b", "winner_model_a", "winner_model_b", "winner_tie"
]
dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=remove_columns,
    num_proc=1,
    batch_size=1000,
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# =======================
# 6. 训练参数
# =======================
print("⚙️ 配置训练参数...")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,  # 根据显存调整
    gradient_accumulation_steps=4,   # 等效 batch_size = 64
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    
    # --- 关键修改：启用 FP16 并开启梯度缩放 ---
    fp16=True,                       # 启用混合精度训练
    # -----------------------------------------

    gradient_checkpointing=True,     # 节省显存
    disable_tqdm=False,
    report_to="none",
    remove_unused_columns=False,     # 避免删除 'label'
)

# =======================
# 7. 初始化 Trainer
# =======================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# =======================
# 9. 开始训练
# =======================
print("🚀 开始微调...")
try:
    trainer.train()
except Exception as e:
    print(f"❌ 训练失败: {e}")
    raise

# =======================
# 10. 保存 LoRA 适配器
# =======================
print("💾 保存微调权重...")
model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapter"))
print(f"✅ 微调完成！模型已保存到 {output_dir}/lora_adapter")