import csv
from math import exp
import random
random.seed(1)

# Calculate logistic
def logistic(x):
    s = 1 / (1 + exp(-x))
    return s

# Calculate dot product of two lists
def dot(x, y):
  s = 0
  for i in range(len(x)):
      s += x[i] * y[i]
  return s

# Calculate prediction based on model
def predict(model, point):
    p = logistic(dot(model, point["features"]))
    return p

# Calculate accuracy of predictions on data
def accuracy(data, predictions):
    correct = 0
    true_true = 0
    true_false = 0
    false_true = 0
    false_false = 0
    for i in range(len(data)):
        if data[i]["label"] == (predictions[i] >= 0.5):
            correct += 1
            if data[i]["label"]:
                true_true += 1
            else:
                false_false += 1
        else:
            if data[i]["label"]:
                true_false += 1
            else:
                false_true += 1
    print("Actual True, Predicted True: ", str(true_true), " Percent: ", str(true_true/len(data)))
    print("Actual True, Predicted False: ", str(true_false), " Percent: ", str(true_false/len(data)))
    print("Actual False, Predicted True: ", str(false_true), " Percent: ", str(false_true/len(data)))
    print("Actual False, Predicted False: ", str(false_false), " Percent: ", str(false_false/len(data)))
    return float(correct)/len(data)

def load_csv(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(line)
    return lines



# Initialize model
def initialize_model(k):
    return [random.gauss(0, 0.5) for x in range(k)]

# Train model using training data
def train(data, epochs, rate, lam):
    model = initialize_model(len(data[0]['features']))
    def parametric(model, point):
      exponent = exp(model[0] + dot(model[1:], point['features'][1:]))
      return exponent/(1+exponent)
    for i in range(epochs):
      updated = []
      for t in range(len(model)):
        sum = 0
        for l in range(len(data)):
          rand = random.randrange(0, len(data))
          sum += data[rand]['features'][t] * (data[rand]['label'] - parametric(model, data[rand]))
        updated.append(model[t] - rate*lam*model[t] + rate*sum)
      model = updated
    return model

# Feature extraction
def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['AD'] == '1')

        features = []
        features.append(1.)
        features.append(float(r['C']))
        point['features'] = features
        data.append(point)
    return data

# Tune your parameters for final submission
def submission(data):
    random.seed(1)
    return train(data, 100, 1e-3, 1e-2)

def test_submission():
    train_data = extract_features(load_csv("training.csv"))
    valid_data = extract_features(load_csv("test.csv"))
    model = submission(train_data)
    predictions = [predict(model, p) for p in train_data]
    print("Training Accuracy:", accuracy(train_data, predictions))
    predictions = [predict(model, p) for p in valid_data]
    print("Validation Accuracy:", accuracy(valid_data, predictions))
test_submission()