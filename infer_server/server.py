from flask import Flask, request, jsonify

from infer import InferClassifier


MODEL_DIR = '../roberta_wwm/roberta_wwm_ext'
CKPT_FILE = '../roberta_wwm/roberta_wwm_ext/final.ckpt'
DEVICE = 'cuda'

app = Flask(__name__)
classifier = InferClassifier(MODEL_DIR, CKPT_FILE, device=DEVICE)

@app.route('/infer_one_class', methods=['GET'])
def infer_one():
    text = request.args.get('text')
    topk = int(request.args.get('topk'))
    return jsonify({'class': classifier.infer_one(text, topk)})


if __name__ == '__main__':
    app.run(debug=True)