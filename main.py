from Util import readBase
from scipy import spatial

shape, rgb, labels = readBase('segmentation.test')

shape_dissimilarity_matrix = spatial.distance.cdist(shape, shape, 'euclidean')
rgb_dissimilarity_matrix = spatial.distance.cdist(rgb, rgb, 'euclidean')

print(shape_dissimilarity_matrix)