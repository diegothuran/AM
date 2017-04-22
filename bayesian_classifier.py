from __future__ import division

import numpy as np
import math

from Util import readBase

shape, rgb, labels = readBase('segmentation.test')
classes = set(labels)

def test(xk, view='shape'):
	_view = []
	offset = 0
	if view == 'shape':
		_view = shape
		offset = 0
	else:
		if view == 'rgb':
			_view = rgb
			offset = 9

	_max = 0
	wl = ''
	for wi in classes:
		score = Posteriori(wi,xk, _view, offset)
		if score > _max:
			_max = score
			wl = wi

	return wl[2:], _max

def Posteriori(wi, xk, view, offset):
	return p(xk, wi, view, offset)*Priori(wi)#/sum([p(xk,wr)*Priori(wr) for wr in classes])

def Priori(wi):
	return len([x for x in labels if x == wi])/len(labels)

def p(xk, wi, view, offset):
	means, covariances = estimate(wi, view)
	d = len(view[0])
	val = ( (2*math.pi)**(-d/2) * 
		np.prod(np.array([(1+covariances[j]) for j in range(0,d)]))**(-1/2) *
		math.exp((-1/2)*sum([(xk[j+offset] - means[j])**2/(1+covariances[j]) for j in range(0,d)]))
	)
	return val

def estimate(wi, view):
	d = len(view[0])
	means = []
	covariances = []
	for j in range(0,d):
		xj = column(viewclass(wi, view),j)
		mean = sum(xj)/len(xj)
		covariance = sum([(xj[k] - mean)**2 for k in range(0,len(xj))])/len(xj)
		means.append(mean)
		covariances.append(covariance)

	return means, covariances

def column(matrix, i):
    return [row[i] for row in matrix]

def viewclass(wi, view):
	subview = []
	for k in range(0,len(view)):
		if labels[k] == wi:
			subview.append(view[k])

	return subview