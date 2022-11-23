import json
import numpy as np
import re

path = "annotations.txt"
path_out = "dataset.json"


class LazyDecoder(json.JSONDecoder):
	def raw_decode(self, s, **kwargs):
		regex_replacements = [
			(re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),
			(re.compile(r',(\s*])'), r'\1'),
		]
		for regex, replacement in regex_replacements:
			s = regex.sub(replacement, s)
		return super().raw_decode(s, **kwargs)


def convert(line):
	res = ''
	while '{"text' in line:
		line = line[line.find('{"text'):]
		r = line.find('","meta')
		curr = line[:r] + '",'

		# line = line[r + 2:]
		line = line[line.find('"tokens":['):]
		r = line.find('],')
		curr += line[:r] + '],'

		if '"spans":[' in line and line.find('"spans":[') < line.find('"answer":'):
			line = line[line.find('"spans":['):]
			r = line.find('],')
			curr += line[:r] + '],'
			line = line[r:]

		line = line[line.find('"answer":'):]
		r = 17
		curr += line[:r] + '}'

		res += curr + ' '
		line = line[18:]

	return res


def make_json_list(line):
	dec = LazyDecoder()
	res = []
	# line = convert(line)
	while '{"text' in line:
		line = line[line.find('{"text'):]  # removing noise

		json_data, r = dec.raw_decode(line)  # retrieving the first json in line and its right limit
		res.append(json_data)  # adding it to the result
		line = line[r:]  # removing it from the parsed line

	return res

# tags list
NER_TAGS = ['O', 'LOC', 'PERSON', 'LOCATION', 'ORG',
			'GPE', 'LANGUAGE', 'NAT_REL_POL', 'DATETIME',
			'PERIOD', 'QUANTITY', 'MONEY', 'NUMERIC',
			'ORDINAL', 'FACILITY', 'WORK_OF_ART', 'EVENT']

tags_idx = {}
for it in range(len(NER_TAGS)):
	tags_idx[NER_TAGS[it]] = it

final_list = []

it = 0
print(f'Started reading data from {path}')
with open(path, encoding='utf-8') as f:
	for line in f:
		jsons = make_json_list(line)
		for data in jsons:
			if data['answer'] == 'accept':
				l = data['tokens']
				tokens = [x['text'] for x in l]
				spaces = [x['ws'] for x in l]
				ner_tags = np.zeros(len(spaces), int)
				ner_ids = np.zeros(len(spaces), int)

				if 'spans' in data.keys():
					spans = data['spans']

					for span in spans:
						left, right = span['token_start'], span['token_end']
						label = span['label']

						ner_id = tags_idx[label]
						ner_tags[left: right + 1] = ner_id
						ner_ids[left: right + 1] = ner_id

				current = {
					'id': int(it),
					'tokens': tokens,
					'ner_ids': [int(x) for x in ner_ids],
					'space_after': spaces,
					'ner_tags': [NER_TAGS[x] for x in ner_tags]
				}
				final_list.append(current)

				json_obj = json.dumps(current)
				print(it)

				it += 1

print(f'Finished reading data from {path}')
print(f'Saving converted data to {path_out}')
with open(path_out, 'w') as f:
	# print(final_list)
	json_obj = json.dumps(final_list)
	f.write(json_obj)
print(f'Finished saving converted data to {path_out}')




