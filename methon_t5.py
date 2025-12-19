import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载T5模型和Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5_small")
model = T5ForConditionalGeneration.from_pretrained("t5_small")

# 编码输入文本和目标输出
encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
inputs = encodings.input_ids
outputs = torch.tensor(labels)
decoder_input_ids = torch.tensor([[tokenizer.pad_token_id if i != 0 else tokenizer.eos_token_id for i in output] for output in outputs.view(-1, 1)])

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 进行有监督预训练
model.train()
for epoch in range(3):  # 进行3轮训练
    optimizer.zero_grad()
    logits = model(inputs, labels=decoder_input_ids)[1]
    loss = torch.nn.CrossEntropyLoss()(logits.view(-1, model.config.vocab_size), outputs.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# 使用预训练模型进行文本分类
model.eval()
with torch.no_grad():
    test_texts = ["这是一个正面的例子。"]
    test_encodings = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
    test_inputs = test_encodings.input_ids
    test_decoder_inputs = test_encodings.input_ids  # 使用相同的输入作为decoder输入
    test_logits = model(test_inputs, decoder_input_ids=test_decoder_inputs)[0]
    test_preds = test_logits.argmax(dim=1).tolist()
    print("预测结果：", test_preds)

