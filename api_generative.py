from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = Flask(__name__)
CORS(app)

# โหลดโมเดลแบบ generative ที่ฝึกเสร็จแล้ว
model_path = "./best_generative_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# สร้าง pipeline สำหรับ generative
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route('/answer', methods=['POST'])
def answer_question():
    data = request.get_json()
    context = data.get('context')

    if not context:
        return jsonify({'answer': "กรุณาใส่ context สำหรับการถาม"})

    # ใช้ pipeline ในการสร้างคำตอบ
    prompt = f"Context: {context}\nAnswer:"
    try:
        result = qa_pipeline(prompt, max_length=150, num_return_sequences=1)
        generated_text = result[0]["generated_text"]

        # แยกคำตอบจากข้อความที่โมเดลสร้าง
        answer = generated_text.split("Answer:")[-1].strip()

        if not answer:
            answer = "ไม่สามารถให้คำตอบได้จาก context ที่ให้มา"
        
    except Exception as e:
        answer = "เกิดข้อผิดพลาดในการหาคำตอบ"

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(port=5000)
