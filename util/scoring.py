def score(predicted, truth):
	assert(len(predicted) == len(truth))

	nr_correct = 0.0
	for i in range(0, len(predicted)):
		if predicted[i] == truth[i]:
			nr_correct += 1.0
	return nr_correct / len(predicted)