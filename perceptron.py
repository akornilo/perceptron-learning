from collections import Counter
from random import shuffle

#scikit_learn
category = "atheistchristian"
keyFile = open("data/" + category + ".response")

key = {}

# Get the labels for all the articles
for line in keyFile:
	line = line.strip() # Cleanup
	line = line.split() # Break apart on white space

	# Convert 0 label to -1
	if int(line[1]) == 0:
		key[line[0]] = -1
	else:
		key[line[0]] = 1

# Helper method for data point files
def parseDataFile(file):
	data = {}
	f = open(file)
	for line in f:
		line = line.strip()

		split = line.index("{") # Find split point
		article = line[:split - 1]

		vec = eval(line[split:])
		data[article] = vec
	return data


trainFile = "data/" + category + ".train"

trainData = parseDataFile(trainFile).items()

# Helper functions for vectors
def dotProd(v1, v2):
	total = 0
	for key, val in v1.items():
		total += val * v2.get(key, 0)

	return total

def vecAdd(v1, v2, sign = 1):
	for key, val in v2.items():
		v1[key] = v1.get(key, 0) + sign * val
	return v1


# Start of PLA algorithm
weights = {}

# Pick a fixed number of times to iterate through the data
for i in range(10):
	# Shuffle points to avoid bias in presentation
	shuffle(trainData)

	for article, vec in trainData:

		dp = dotProd(weights, vec)

		actualSign = key[article]

		# Check if article was misclassified
		if not (dp * actualSign > 0):
			weights = vecAdd(weights, vec, actualSign)

# Evaluation code
def evalData(weights, points):
	global key
	wrong = 0

	for article, vec in points:

		dp = dotProd(weights, vec)

		actualSign = key[article]

		# Check if article was misclassified
		if not (dp * actualSign > 0):
			wrong += 1
	return wrong

wrong = evalData(weights, trainData)
total = len(trainData)
print "Train Data:", wrong, total, wrong * 100.0 / total

testData = parseDataFile("data/" + category + ".test").items()
wrong = evalData(weights, testData)
total = len(testData)
print "Test data:", wrong, total, wrong * 100.0 / total
