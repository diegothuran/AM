from Util import readBase, generate_dissimilarity_matrix
from scipy import spatial

#variaveis que recebem os valores correspondentes pada shape as 9 primeiras
# rgb as 10 ultimas
# labels as classificacoes
shape, rgb, labels = readBase('segmentation.test')

diss = generate_dissimilarity_matrix(shape.tolist())

print(diss)