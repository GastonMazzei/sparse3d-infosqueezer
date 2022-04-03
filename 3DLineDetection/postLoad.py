import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

if 0 == os.path.getsize("out.txtlines.txt"):
	try:
		os.system(f'rm -r segmentedLines')
		os.system('mkdir segmentedLines')
	except: pass
	sys.exit(0)

# Define verbosity
DISPLAY = [False, True][0]

# Open files
if True:
	df = pd.read_csv('out.txtlines.txt', sep='   ', engine='python')
elif False:
	with open('out.txtlines.txt','r') as f:
		s = f.readlines()
	cols = ['x','y','z','slopeX','slopeY','slopeZ','id']
	df = {x:[] for x in cols}
	for line in s:
		localdata = line.split('   ')
		for i,k in enumerate(cols):
			df[k].append(float(localdata[i]))
	df = pd.DataFrame(df)
elif False:
	with open('out.txtlines.txt','r') as f:
		s = f.read()
	s = s.replace('   ',',')
	with open('postprocessed.out.txtlines.txt','w') as f:
		f.write(s)	
	df = pd.read_csv('postprocessed.out.txtlines.txt')

# Compute  ids
ids = list(set(df.iloc[:,-1]))

# Show number of detected lines
print(f'ids are: {ids}')

# Split data in lines
lines = {}
for ID in ids:
	local_data = df[df.iloc[:,-1] == ID]
	lines[ID] = local_data.iloc[:,:3].to_numpy().copy()

# Display
if DISPLAY:
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	for ID in ids:
		ax.scatter(lines[ID][:,0],lines[ID][:,1],lines[ID][:,2], label=str(ID))
	plt.show()

# Record
try:
	os.system(f'rm -r segmentedLines')
	os.system('mkdir segmentedLines')
except: pass
c=0
for ID in ids:
	pcd_out = o3d.geometry.PointCloud()
	pcd_out.points = o3d.utility.Vector3dVector(lines[ID])
	o3d.io.write_point_cloud(f"segmentedLines/segmented_line{c}.ply", pcd_out)
	c+=1
