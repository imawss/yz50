input = 0.25
weight = 0.5
bias = 10

output = max((input * weight) + bias, 0)
print(output)

#sigmoid func yerine ReLU olması için max ile output.