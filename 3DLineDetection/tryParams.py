import os, sys
import matplotlib.pyplot as plt
import numpy as np


all_a = [10,25,50,100] # Number of neighors
all_b = [10,25,50,75] # Angle
all_c = [0.75,1,1.25]
all_d = [40,50,60]


# Process for different values of A and B
def process(d,c):
	global all_a, all_b
	numbers = []
	for a in all_a:
		for b in all_b:
			os.system(f'src/LineFromPointCloud {a} {b} {c} {d} > tmp')
			with open('tmp','r') as file:
				try:
					_number = int([x for x in file.readlines() if 'lines number:' in x][0].split(':')[1])
					numbers += [_number].copy()
				except: pass
				try:
					os.system(f'rm tmp')
				except: pass
	return np.asarray(numbers)


if len(sys.argv)<2:
	# Compute for different values of C and D
	results = []
	fig, ax = plt.subplots(len(all_c),len(all_d),figsize=(30,30))
	for i,c in enumerate(all_c):
		for j,d in enumerate(all_d):
			results += [process(d,c)]


	# Display
	MAX = max([np.max(x.flatten()) for x in results])
	for i in range(len(results)):
		I,J = i // len(all_d), i % len(all_d)
		im = ax[I,J].imshow(results[i].reshape((len(all_a), len(all_b))), vmin=0, vmax=MAX)
		ax[I,J].set_title(f'{i+1}')
		plt.colorbar(im, ax=ax[I,J])
	plt.show()


	# Collect optimal value from user
	i,j,k = [int(x) for x in input('Please write the comma-separated ixes, starting w the vertical dimension, and finally the image from 1 to 9. E.g. "2,3,3"').split(',')]
	print(f'Chosen i,j,k are {i} {j} {k}')
else:
	i,j,k = [int(x) for x in sys.argv[1:]]
k-=1
A = all_a[i]
B = all_b[j]
Cix,Dix = k // len(all_d), k % len(all_d)
C,D = all_c[Cix], all_d[Dix]
os.system(f'src/LineFromPointCloud {A} {B} {C} {D}')
	
