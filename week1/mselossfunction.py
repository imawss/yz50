#MSE Loss Function denemesi
inputs = [0.1, 0.75, 0.42]

weights = [
    [9, 12, 34],
    [10, 3, 0.86],
    [10, 3, 0.24]
]

bias = [0, 2, 6]

output_activations = [0, 0, 0]
samples = [24, 6, 10]

for i,j in enumerate(output_activations):
    weighted_sum = float()

    for k,l in enumerate(inputs):
        weighted_sum += weights[i][k] * l
    output_activations[i] = weighted_sum + bias[i] #Aktivasyon kaldırdım aşağıda notu var

loss = float()

for i,j in enumerate(output_activations):
    loss += pow((samples[i]- j), 2) / len(output_activations)

print(output_activations)
print(loss)

#Son Ölçülen Loss: 0.201

#Videoda hep pozitif değer alınması gerektiği 
# ve birden fazla katman olduğu için ReLU kullanılmış 
# ancak burada -bias ile nöronun ölmesine sebep oluyor.
#TODO: Kaldırılması daha iyi olur çünkü 0'a eşitlediği için hatanın kaç
#olduğu anlaşılamıyor.