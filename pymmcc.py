# python lib to compute MMCC adjust

import pytopo as pt
import numpy as np
import pandas as pd
from math import pi

def coordinates(name, ne_fix, ne_approx):
	'''
	returns east, north for stn.
	search in x (fix) pr in x0 (aprox).
	'''
	if name in ne_fix.index:
		df = ne_fix
	elif name in ne_approx.index:
		df = ne_approx
	else:
		return [None, None]
	return [df.loc[name,'East'], df.loc[name,'North']]

def join_observation(dh, ang):
	'''
	returns tuple (measure, values) from dh and ang (both are panda DataFrame).
	measure is a tuple list were each element is a measure label.
	values is a numpy column array were each elemen is a value of measure.
	'''
	measure = []
	value = []
	for i in range(len(dh)):
		value.append(dh.loc[i, 'Value'])
		measure.append((dh.loc[i,'From'], dh.loc[i,'To']))
	for i in range(len(ang)):
		value.append(ang.loc[i, 'Value'])
		measure.append((ang.loc[i,'Backsight'], ang.loc[i,'Station'], ang.loc[i,'Foresight']))
	return measure, np.array([value]).T

def approximate_values(station, fsAz_0, dh, angle):
	'''
	return ne (panda DataFrame) with the aproximates values computes from
	observations (distances and angles). Firs two parameters are station coordinates 
	and foresight azimth (i-1), both parameters for propagate coordinates of traverse.
	'''
	stn = list(angle.iloc[1:len(angle)-1, 1])
	ne = pd.DataFrame(columns=['North', 'East'], dtype='float64')
	ne.index.name = 'Stn'
	for i in range(len(stn)):
		fsAz_ij = pt.foresight(fsAz_0, pt.grad2rad(angle.loc[i, 'Value']))
		dN, dE = pt.dNorth(fsAz_ij, dh.loc[i, 'Value']), pt.dEast(fsAz_ij, dh.loc[i, 'Value'])
		station = [station[0] + dE, station[1] + dN]
		ne.loc[stn[i]]= {'North': station[1], 'East': station[0]}
		fsAz_0 = fsAz_ij
	return ne

def jacobian(measure, ne_fix, ne_approx):
	'''
	returns jacobian matrix from the measures, fixed coordinates and aproximates coordinates
	for each unknown.
	'''
    j = np.zeros((len(measure), 2*len(ne_approx)))
    for l in range(len(j)):
        stns = measure[l]
        # coordinate by measure
        xy = []
        for stn in stns:
            xy.append(coordinates(stn, ne_fix, ne_approx))
        if len(xy) == 2: 
            # distances
            coef = linealeq_distance(xy[0], xy[1])
        else: 
            # angles
            coef = pt.rad2grad(linealeq_angle(xy[0], xy[1], xy[2]))
            
        # fill jacobian matrix
        for c in range(len(coef)):
            
            if stns[c] in ne_approx.index:
                i = list(ne_approx.index).index(stns[c])
                j[l][i*2] = coef[c][0]   #dx
                j[l][i*2+1] = coef[c][1] #dy
    return j

def corrections(j, k, w):
	'''
	returns a tuple of N matrix and X vector for the adjust.
	'''
	n = np.dot(np.dot(j.T, w), j)
	jtwk = np.dot(np.dot(j.T, w), k)
	x = np.dot(np.linalg.inv(n), jtwk)
	return n, x

def ne_adjusted(ne_approx, dxs):
	'''
	returns unknown adjusted.
	'''
	stn = ne_approx.index
	ne = pd.DataFrame(columns=['North', 'East'], dtype='float64')
	ne.index.name = 'Stn'
	for i in range(len(stn)):
		ne.loc[stn[i],'East'] = ne_approx.loc[stn[i],'East'] + dxs[i*2]
		ne.loc[stn[i],'North'] = ne_approx.loc[stn[i],'North'] + dxs[i*2+1]
	return ne

def observations(measure, ne_fix, ne_approx, p='rad'):
	'''
	returns numpy array with computes observacions (Lb vector).
	if p=0 rad,
	if p=1 grad,
	if p=2 deg
	'''
	if p == 'rad':
		c = 1
	elif p == 'grad':
		c = 200 / pi
	elif p == 'deg':
		c = 180 / pi
	value = np.zeros(len(measure))
	for i in range(len(value)):
		stns = measure[i]
		xy = np.array(np.zeros([len(stns), 2]))
		for stn in range(len(xy)):
			xy[stn] = coordinates(stns[stn], ne_fix, ne_approx)

		if len(xy) == 2: 
			# distances
			value[i] = pt.distance(xy[0], xy[1])
		else: 
			# angles
			value[i] = pt.angle(xy[0], xy[1], xy[2]) * c
	return np.array(([value])).T

def linealeq_distance(i, j):
	'''
	return coefs for the jacobian matrix.
	This are the implementation of distance lineal observation equation.
	'''
	cdxi = (i[0] - j[0]) / pt.distance(i,j)
	cdyi = (i[1] - j[1]) / pt.distance(i,j)
	cdxj = (j[0] - i[0]) / pt.distance(i,j)
	cdyj = (j[1] - i[1]) / pt.distance(i,j)
	return np.array([[cdxi, cdyi], [cdxj, cdyj]])

def linealeq_angle(b, i, f):
	'''
	return coefs for the jacobian matrix.
	This are the implementation of angle lineal observation equation.
	'''
	cdxb = (i[1] - b[1]) / pt.distance(i,b)**2
	cdyb = (b[0] - i[0]) / pt.distance(i,b)**2
	cdxi = (b[1] - i[1]) / pt.distance(i,b)**2 - (f[1] - i[1]) / pt.distance(i,f)**2
	cdyi = (i[0] - b[0]) / pt.distance(i,b)**2 - (i[0] - f[0]) / pt.distance(i,f)**2
	cdxf = (f[1] - i[1]) / pt.distance(i,f)**2
	cdyf = (i[0] - f[0]) / pt.distance(i,f)**2
	return np.array([[cdxb, cdyb], [cdxi, cdyi], [cdxf, cdyf]])