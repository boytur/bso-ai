from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset

# กำหนดชื่อโมเดลที่จะใช้
model_name = "distilbert-base-uncased"

# โหลด Tokenizer และ Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# โหลด dataset
dataset = load_dataset("json", data_files="blog_data.json")

# ฟังก์ชันเตรียมข้อมูล
def preprocess_function(examples):
    # Tokenize inputs และ labels
    inputs = tokenizer(examples['question'], examples['context'], truncation=True, padding="max_length", max_length=512)
    start_positions = []
    end_positions = []

    for i in range(len(examples["context"])):
        context = examples["context"][i]
        answer = examples["answer"][i]
        start_idx = context.find(answer)
        if start_idx == -1:
            start_idx = 0
        end_idx = start_idx + len(answer)
        start_positions.append(start_idx)
        end_positions.append(end_idx)

    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions

    return inputs

# เตรียมข้อมูล
tokenized_data = dataset["train"].map(preprocess_function, batched=True)

# ตั้งค่าการฝึก
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=100,
)

# สร้าง Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    eval_dataset=tokenized_data
)

# เริ่มการฝึก
trainer.train()

# บันทึกโมเดลที่ฝึกเสร็จแล้ว
model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")
