inputs = [0.1, 0.75, 0.42]

weights1 = [
    [9, 12, 34],
    [10, 3, 0.86],
    [10, 3, 0.24]
]

bias1 = [0, 2, 6]

samples = [24, 6, 10]

h = 0.0001
learning_rate = 0.1
epochs = 200


def loss(weights, bias):
    output_activations = [0, 0, 0]

    for i in range(len(output_activations)):
        weighted_sum = 0.0

        for k, l in enumerate(inputs):
            weighted_sum += weights[i][k] * l

        output_activations[i] = weighted_sum + bias[i]

    total = 0.0

    for i, j in enumerate(output_activations):
        total += pow((samples[i] - j), 2) / len(output_activations)

    return total


def gradients(weights, bias):
    base = loss(weights, bias)

    weight_grads = [[0.0, 0.0, 0.0] for _ in weights]
    bias_grads = [0.0 for _ in bias]

    for i in range(len(weights)):
        for k in range(len(weights[i])):
            weights[i][k] += h
            weight_grads[i][k] = (loss(weights, bias) - base) / h
            weights[i][k] -= h

    for i in range(len(bias)):
        bias[i] += h
        bias_grads[i] = (loss(weights, bias) - base) / h
        bias[i] -= h

    return weight_grads, bias_grads


print("ilk loss:", loss(weights1, bias1))

for epoch in range(epochs):
    weight_grads, bias_grads = gradients(weights1, bias1)

    for i in range(len(weights1)):
        for k in range(len(weights1[i])):
            weights1[i][k] -= learning_rate * weight_grads[i][k]

    for i in range(len(bias1)):
        bias1[i] -= learning_rate * bias_grads[i]

    if epoch % 20 == 0: # SADECE GÖRSELLEŞTİRME İÇİN EKLEDİM LEARNING RATE TEST EDERKEN YARDIMCI OLDU
        print(epoch, loss(weights1, bias1))

print("son loss:", loss(weights1, bias1))
