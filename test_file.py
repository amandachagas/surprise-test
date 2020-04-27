import json
import random

for i in range(0,5):
	result = dict()
	result['standard_recs'] = random.sample(range(1,100),5)
	result['greedy_recs'] = random.sample(range(1,100),5)
	result['random_recs'] = random.sample(range(1,100),5)
	result['standard_ild'] = i
	result['greedy_ild'] = i
	result['random_ild'] = i
	result['standard_p@3_5_10'] = random.sample(range(0,10), 3)
	result['greedy_p@3_5_10'] = random.sample(range(0,10), 3)
	result['random_p@3_5_10'] = random.sample(range(0,10), 3)
	if i is 0:
		with open("test_file.json", "w") as json_file:
			json_file.write("{}\n".format(json.dumps(result)))
	else:
		with open("test_file.json", "a") as json_file:
			json_file.write("{}\n".format(json.dumps(result)))