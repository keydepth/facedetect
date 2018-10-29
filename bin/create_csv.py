#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import csv


def create(result):
	rank = result['rank']

	i = 0
	input_data = []
	header = ['#rank', 'Job', 'Accuracy']

	for r in rank:
		i += 1
		input_data.append((str(i), r['job'], r['accuracy']))

	# input_data = [(str(i), r['job'], r['accuracy']) for r in rank]
	with open('Result.csv', 'w') as f:
		writer = csv.writer(f, lineterminator='\n')

		writer.writerow(header)
		writer.writerows(input_data)


if __name__ == '__main__':
	result = [
		{
			'no': '13',
			'job': 'denso',
			'accuracy': '0.24'
		},
		{
			'no': '24',
			'job': 'president',
			'accuracy': '0.22'
		}
	]
	result = {
		'rank': result
	}

	create(result)
