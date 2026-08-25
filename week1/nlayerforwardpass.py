#Tek katman, 3 girdi, 3 nöron forward pass denemsi
inputs = [0.1, 0.75, 0.42]

weights = [
    [0.2, 1, 0.35],
    [0.12, 0.20, 0.86],
    [0.76, 0.31, 0.24]
]

bias = [0, 2, -6]

output_activations = [0, 0, 0]

for i,j in enumerate(output_activations):
    weighted_sum = float()

    for k,l in enumerate(inputs):
        weighted_sum += weights[i][k] * l
    output_activations[i] = max(weighted_sum + bias[i], 0)

print(output_activations)
