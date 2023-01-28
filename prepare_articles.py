
import json

with open('good_questions.json') as f:
	data = json.load(f)

for paragraph_num, paragraph in enumerate(data['data'][0]['paragraphs']):
	with open('articles/article_{}.txt'.format(paragraph_num+1), 'w') as f:
		f.write(paragraph['context'])
	