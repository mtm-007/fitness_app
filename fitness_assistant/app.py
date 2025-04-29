import uuid
from flask import Flask, request, jsonify
from rag import rag

import db 

app = Flask(__name__)

@app.route('/question', methods = ['POST'])
def handle_question():
    data = request.json
    question = data['question']

    if not question:
        return jsonify({'error': "No question provided"}), 400
    
    conversation_id = str(uuid.uuid4())

    answer_data = rag(question)

    
    result = {
        "conversation_id" : conversation_id,
        "question" : question,
        "answer" : answer_data["answer"],
        }

    return jsonify(result)

    db.save_conversation(
        conversation_id=conversation_id,
        question=question,
        answer_data=answer_data,
    )

@app.route('/feedback', methods = ['POST'])
def handle_feedback():
    data = request.json
    conversation_id = data["conversation_id"]
    feedback = data["feedback"]

    if not conversation_id or feedback not in [1, -1]:
        return jsonify({"error" : "Invalid input"}), 400
    
    db.save_feedback(
        conversation_id=conversation_id,
        feedback=feedback,
    )
    
    result = {"message" : f"feedback recieved for conversation {conversation_id}: {feedback}"}
   
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug = True)