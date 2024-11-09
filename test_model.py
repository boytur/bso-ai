# test_model.py
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# โหลดโมเดลและ tokenizer ที่ฝึกแล้ว
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# ใช้โมเดลสำหรับถามตอบ
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# ตัวอย่างข้อมูล
context = "การใช้ Hugging Face Transformers ทำให้คุณสามารถฝึกและใช้โมเดล AI ได้ง่ายขึ้น ด้วย API ที่สะดวกและมีโมเดลที่หลากหลาย"
question = "การใช้ Hugging Face Transformers มีประโยชน์อย่างไร?"

# ทดสอบ
answer = qa_pipeline(question=question, context=context)
print("Answer:", answer["answer"])
