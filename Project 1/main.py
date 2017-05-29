from Util import readBase
from scipy import spatial

#variaveis que recebem os valores correspondentes pada shape as 9 primeiras
# rgb as 10 ultimas
# labels as classificacoes
shape, rgb, labels = readBase('segmentation.test')

#matrix de dissimilaridade para as variaveis do shape
shape_dissimilarity_matrix = spatial.distance.cdist(shape, shape, 'euclidean')
#matrix de dissimilaridade para as variaveis rgb
rgb_dissimilarity_matrix = spatial.distance.cdist(rgb, rgb, 'euclidean')

print(shape_dissimilarity_matrix)