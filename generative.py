import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from datasets import Dataset



print(torch.__version__)
print(torch.cuda.is_available())

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# ฟังก์ชันคำนวณความคล้ายคลึง (Cosine Similarity)
def calculate_similarity(predicted_answer, true_answer):
    vectorizer = TfidfVectorizer().fit_transform([predicted_answer, true_answer])
    similarity_matrix = cosine_similarity(vectorizer)
    return similarity_matrix[0, 1]

# กำหนดชื่อโมเดลที่เป็น generative เช่น GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# กำหนด pad_token ให้เป็น eos_token (อาจจะจำเป็นในการฝึกโมเดล generative)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# ตรวจสอบว่า GPU มีหรือไม่ ถ้ามีให้ใช้ GPU ถ้าไม่มีให้ใช้ CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# แสดงว่าใช้ GPU หรือ CPU
if torch.cuda.is_available():
    print("Using GPU for training")
else:
    print("Using CPU for training")

# โหลดข้อมูลจากไฟล์ JSON
with open('D:/bso/ai/blog_data.json', 'r') as f:
    data = json.load(f)

# แปลงข้อมูลเป็น Dataset
dataset = Dataset.from_dict({
    "context": [item["context"] for item in data],
    "answer": [item["answer"] for item in data]
})

# ตั้งค่าการ log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ฟังก์ชันเตรียมข้อมูล
def preprocess_function(examples):
    inputs = [f"Context: {context}" for context in examples["context"]]
    outputs = [answer for answer in examples["answer"]]

    # Tokenize inputs และ outputs
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    tokenized_outputs = tokenizer(outputs, padding="max_length", truncation=True, max_length=512)

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_outputs["input_ids"]
    }

# เตรียมข้อมูล
tokenized_data = dataset.map(preprocess_function, batched=True)

# ตั้งค่าการฝึก
training_args = TrainingArguments(
    output_dir="./results",                    # โฟลเดอร์สำหรับเก็บผลลัพธ์
    per_device_train_batch_size=2,             # ขนาดของ batch ในการฝึก
    per_device_eval_batch_size=2,              # ขนาดของ batch สำหรับการประเมิน
    num_train_epochs=3,                        # จำนวน epochs ที่ต้องการฝึก
    save_steps=500,                            # การบันทึกโมเดลทุก 500 steps
    save_total_limit=2,                        # จำกัดจำนวนโมเดลที่บันทึก
    eval_strategy="epoch",                     # ประเมินผลทุก epoch
    logging_dir='./logs',                      # โฟลเดอร์สำหรับเก็บ logs
    logging_steps=100,                         # log ทุกๆ 100 steps
    no_cuda=False                              # ใช้ GPU ถ้ามี
)

# สร้าง Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    eval_dataset=tokenized_data,               # ใช้ dataset เดียวกันเพื่อทดสอบง่าย ๆ
)

# การฝึกซ้ำจนกว่าจะได้คำตอบที่มีความคล้ายคลึง 90%
target_similarity = 0.90
best_similarity = 0.0
while best_similarity < target_similarity:
    logger.info("Training in progress...")

    # เริ่มการฝึก
    trainer.train()

    # ทดสอบโมเดลโดยการคำนวณความคล้ายคลึงระหว่างคำตอบที่โมเดลสร้างและคำตอบที่แท้จริง
    predictions = []
    true_answers = []
    for i in range(len(tokenized_data)):
        context = tokenized_data[i]["input_ids"]
        true_answer = tokenized_data[i]["labels"]

        # สร้างคำตอบจากโมเดล
        input_text = tokenizer.decode(context, skip_special_tokens=True)
        generated_answer = model.generate(tokenizer.encode(input_text, return_tensors="pt").to(device), max_new_tokens=100)

        # ดึงคำตอบที่โมเดลสร้าง
        decoded_answer = tokenizer.decode(generated_answer[0], skip_special_tokens=True)
        
        # เก็บคำตอบที่แท้จริงและคำตอบที่โมเดลสร้าง
        predictions.append(decoded_answer)
        true_answers.append(tokenizer.decode(true_answer, skip_special_tokens=True))

    # คำนวณความคล้ายคลึงเฉลี่ย
    similarities = [calculate_similarity(pred, true) for pred, true in zip(predictions, true_answers)]
    avg_similarity = np.mean(similarities)
    logger.info(f"Average similarity: {avg_similarity}")
    print(f"Average similarity after training round: {avg_similarity:.2f}")

    # ถ้า similarity สูงกว่า 90% ให้หยุดฝึก
    if avg_similarity >= target_similarity:
        best_similarity = avg_similarity
        logger.info(f"Training complete. Best similarity: {best_similarity}")
        print(f"Training complete with best similarity: {best_similarity:.2f}")
        break

# บันทึกโมเดลที่ฝึกเสร็จแล้ว
model.save_pretrained("./best_generative_model")
tokenizer.save_pretrained("./best_generative_model")
