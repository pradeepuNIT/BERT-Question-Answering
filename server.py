
import tensorflow as tf
import json
from flask import Flask, json, request, jsonify, make_response, render_template
from just_ask import Chat as chat_on_csv
from ask_on_paragraph import main
import io
import csv

from models_config import MODELS_CONFIG

app = Flask(__name__)

chat_models = {}

def load_model():
	global chat_models
	for model, params in MODELS_CONFIG["models"].items():
		chat_models[model] = main(
			MODELS_CONFIG['config_file'], params['bert_architecture'],
			params['version'], params['model_number'], n_best_size=20)


@app.route('/')
def api_root():
	return render_template('index.html')


@app.route('/read_csv', methods=["POST"])
def read_csv():
	data = {'success': False}
	f = request.files.get('file')
	if not f:
		return "No file"
	stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
	csv_input = csv.reader(stream)
	data['data'] = list(csv_input)[1:]
	data['success'] = True
	return data


@app.route('/read_content_file', methods=["POST"])
def read_content_file():
	data = {'success': False}
	f = request.files.get('content_file')
	if not f:
		return data
	content = f.read().decode("utf-8")
	data['data'] = content.split('\n')
	data['success'] = True
	return data


@app.route('/process_question', methods=['POST'])
def process_question():
	data = {'success': False}
	if request.method == "POST":
		model = request.form.get('model')

		data = json.loads(request.form.get('data'))
		contexts = data['paragraphs']
		titles = data['titles']

		skip_paraselection = data.get('skip_paraselection')
		top_para_count = data.get('top_para_count')
		context_by_titles = dict(zip(titles, contexts))

		question = request.form.get('question')

		chat = chat_models[model]
		chat.digest_contexts(
			context_by_titles, top_para_count=top_para_count,
			skip_selection=skip_paraselection)
		predictions = chat.get_answers(question)

		data['predictions'] = predictions
		data['success'] = True
	print("Predictions: {}".format(predictions))
	return jsonify(data)


if __name__ == '__main__':
	load_model()
	app.run(debug=True)

'''
curl -i -H "Content-Type: application/json" -X POST -d \
	'{"model":"model1", "question": "What is advising bank?"}' http://localhost:5000/question_para
'''
