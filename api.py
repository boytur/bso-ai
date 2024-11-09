from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

app = Flask(__name__)
CORS(app)

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model_path = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# สร้าง pipeline สำหรับถาม-ตอบ
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

@app.route('/answer', methods=['POST'])
def answer_question():
    data = request.get_json()
    question = data.get('question')
    context = data.get('context')

    if not question:
        return jsonify({'answer': "กรุณาใส่คำถาม"})

    if not context:
        return jsonify({'answer': "กรุณาใส่ context"})

    try:
        result = qa_pipeline(question=question, context=context)
        answer = result['answer']
    except Exception as e:
        answer = "เกิดข้อผิดพลาดในการหาคำตอบ"

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(port=5000)
