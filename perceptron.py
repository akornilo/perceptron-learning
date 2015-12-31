from collections import Counter
from random import shuffle

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

trainData = parseDataFile(trainFile)

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

weights = {}

# Small hack to simplify shuffle
trainData = trainData.items()

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

print Counter(weights)
# Evaluate on the training data
wrong = 0
total = 0

for article, vec in trainData:

	dp = dotProd(weights, vec)

	actualSign = key[article]

	# Check if article was misclassified
	if not (dp * actualSign > 0):
		wrong += 1
	total += 1

print "Train Data:", wrong, total, wrong * 100.0 / total

# Evaluate on the test set

testData = parseDataFile("data/" + category + ".test")
# Evaluate on the training data
wrong = 0
total = 0

for article, vec in testData.items():

	dp = dotProd(weights, vec)

	actualSign = key[article]

	# Check if article was misclassified
	if not (dp * actualSign > 0):
		wrong += 1
	total += 1

print "Test data:", wrong, total, wrong * 100.0 / total