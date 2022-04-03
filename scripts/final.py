import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from numpy import linalg as LA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, sys
from sklearn.preprocessing import MinMaxScaler as mms
import plotly.express as px
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from segmentHelper import (
			straighten_pcd,
			twist_back,
			twist_back_meshObject,
			straighten_meshObj_fromOp,
				)


# Define 'smart' operations to perform
JOIN_TRANSVERSED_LINES = [False,True][1] # Join objects transversed by detected lines
TRANSVERSED_LINES_ON_BOX = [False,True][1] # Compute the transversed lines on the box (instead of the hull)
TRANSVERSED_LINES_OUTPUT_HULL = [False,True][0] # Dont compute the convex hull of the objects joined by lines
JOIN_CLOSE_ONES = [False,True][0] # Dont join close ones


# Define containers
plane_information = {}
lines = {}
clusters = {}

# Make an empty directory called objects
try:
	os.system('rm -r objects')
except: pass
try:
	os.mkdir('objects')
except: pass

# Open (real, hardware-computed) planes
for i,filename in enumerate(os.listdir('raw_data/real_planes/')):
	if filename[-3:]=='obj':
		plane_information[i] = o3d.io.read_triangle_mesh('raw_data/real_planes/' + filename).compute_convex_hull()[0]


# Open (estimated, software-computed) clusters
for i,filename in enumerate(os.listdir('segmented/')):
	if filename[-3:]=='ply':
		clusters[i] = o3d.io.read_point_cloud('segmented/' + filename).compute_convex_hull()[0]


# Open (estimated, software-computed) lines
for i,filename in enumerate(os.listdir('3DLineDetection/segmentedLines/')):
	if filename[-3:]=='ply':
		temp_pcd = o3d.io.read_point_cloud('3DLineDetection/segmentedLines/' + filename)
		# Give some bulk so that the 3D convex hull can be computed
		temp_pcd_points = np.asarray(temp_pcd.points)
		temp_pcd_points = np.concatenate([temp_pcd_points, temp_pcd_points+0.05*np.random.rand(*temp_pcd_points.shape)], axis=0)
		temp_pcd = o3d.geometry.PointCloud()
		temp_pcd.points = o3d.utility.Vector3dVector(temp_pcd_points)
		lines[i] = temp_pcd.compute_convex_hull()[0]


#****************************************************************************************
#*********************************Start Applying Filters*********************************
#****************************************************************************************

# Filter 1) do not use real detected planes that are just part of the floor
# Compute the minimum vertical value of the scene, i.e. where is the floor situated :-)
with open('raw_data/boxpoints_info.txt','r') as file:
	info = [float(x) for x in file.readlines()] # info[A] is Ly, and info[B] is the Y-offset
						    # so - info[A] / 2 + info[B] is the floor
						    # A,B = 0,4 apparently, tho it should be 1,4 :O!
FLOOR_TOL = 0.1 #10cm
Ymin = - info[0] / 2 + info[4]
Ymax = info[0] / 2 + info[4]
floor_value = Ymin + FLOOR_TOL # we add 10cm tolerance :-)
ceiling_value = Ymax
print(f'\n{"-"*8}\nFloor\'s value is {floor_value}\n{"-"*8}\n')
for k in list(plane_information.keys()):
	local_val = np.asarray(plane_information[k].vertices)[:,1].max()
	if floor_value > local_val:
		del(plane_information[k]) # Remove the floor-ish planes
		sentence = True
	else: sentence=False
	if sentence: print(f'This planes\' Y min was: {local_val}, sentence={sentence}')




# Collect all meshes in the X,Y,Z aligned version :-)
all_meshes = []
ixes_for_real_planes = []
ixes_for_lines = []
ixes_for_clusters = []
dummy_ixes = []
op = False
counter = 0
for v in plane_information.values():
	local_v, op = straighten_meshObj_fromOp(v,op)
	all_meshes += [local_v]
	ixes_for_real_planes.append(counter)
	counter += 1
for v in lines.values():
	local_v, op = straighten_meshObj_fromOp(v,op)
	all_meshes += [local_v]
	ixes_for_lines.append(counter)
	counter += 1
for v in clusters.values():
	local_v, op = straighten_meshObj_fromOp(v,op)
	all_meshes += [local_v]
	ixes_for_clusters.append(counter)
	counter += 1

# Compute bounding boxes
all_boxes = []
def return_box(mesh):
	"""
	Receives an o3d Triangle Mesh and returns the same 
	type of object but with the information of the (axis aligned) 
	bounding box
	"""
	obb = o3d.geometry.AxisAlignedBoundingBox()
	obb = obb.create_from_points( o3d.utility.Vector3dVector(mesh.vertices) )
	temp = pcd_out = o3d.geometry.PointCloud()
	temp.points = o3d.utility.Vector3dVector(np.asarray( obb.get_box_points() ))
	return temp.compute_convex_hull()[0]
def return_convex_hull(mesh):
	"""
	Receives an o3d Triangle Mesh and returns the same 
	type of object but with the information of the convex hull
	"""
	return mesh.compute_convex_hull()[0]
for mesh in all_meshes:
	all_boxes.append(return_convex_hull(mesh))

# Filter 2) Restrict all the points to the domain Xmin,Xmax,Ymin,Ymax,Zmin,Zmax
FILTER_2 = [False,True][0]
if FILTER_2:
	Xmax = info[1] / 2 + info[3]
	Xmin = -info[1] / 2 + info[3]
	Zmax = (-info[2] / 2 + info[5]) * -1
	Zmin = (info[2] / 2 + info[5]) * -1
	for i in range(len(all_boxes)):
		points = np.asarray(all_boxes[i].vertices)
		rescaled_points = []
		for x in points:
			Xval,Yval,Zval = x
			# Rescale x
			if Xval > Xmax:
				Xval = Xmax
			elif Xval < Xmin:
				Xval = Xmin
			# Rescale x
			if Yval > Ymax:
				Yval = Ymax
			elif Yval < Ymin:
				Yval = Ymin
			# Rescale x
			if Zval > Zmax:
				Zval = Zmax
			elif Zval < Zmin:
				Zval = Zmin
			rescaled_points.append([Xval,Yval,Zval].copy())
		all_boxes[i].vertices = o3d.utility.Vector3dVector(np.asarray(rescaled_points))


# Filter 3) Min accepted vol
MIN_ACCEPTED_VOL = 2e-16 #0.3 * 0.3 * 0.3 # 30 cm x 30 cm x 30 cm
new_all_boxes = []
hist = []
for i,box in enumerate(all_boxes):
	if i in ixes_for_real_planes or i in ixes_for_lines:
		new_all_boxes.append(box)
		continue
	try:
		vol = return_box(box).get_volume()
		hist.append(vol)
	except: vol = np.inf
	if vol > MIN_ACCEPTED_VOL:
		new_all_boxes.append(box)
	else:
		new_all_boxes.append(False)
		dummy_ixes.append(i)
ixes_for_real_planes = [j for j in ixes_for_real_planes if j not in dummy_ixes]
ixes_for_clusters = [j for j in ixes_for_clusters if j not in dummy_ixes]
all_boxes = new_all_boxes
#print(hist)



# *********************AD-HOC 'SMART' OPERATION**************************
# Join together all the boxes that are transversed by detected lines :-)
#************************************************************************
if JOIN_TRANSVERSED_LINES:
	if TRANSVERSED_LINES_ON_BOX:
		all_boxes = [return_box(mesh) if j not in dummy_ixes else False for j,mesh in enumerate(all_boxes)]
	line_intersects = {}
	for i in ixes_for_lines:
		line_intersects[i] = []
		for j,box in enumerate(all_boxes):
			if j in ixes_for_real_planes or j in ixes_for_clusters:
				if i not in dummy_ixes and all_boxes[i].is_intersecting(box):
					line_intersects[i].append(j)
	boxes_intersected_by_lines = []
	for k in line_intersects.keys():
		for i in line_intersects[k]:
			boxes_intersected_by_lines.append(i)
	boxes_not_intersected_by_lines = [i for i in range(len(all_boxes)) if i not in boxes_intersected_by_lines and i not in dummy_ixes]
	print(f'There are {len(boxes_not_intersected_by_lines)} boxes not intersected by lines ({round(100*len(boxes_not_intersected_by_lines)/len(all_boxes),2)})%')
	print(f'There are {len(set(boxes_intersected_by_lines))} boxes not intersected by lines ({round(100*len(set(boxes_intersected_by_lines))/len(all_boxes),2)})%')
	print(f'There are {len(boxes_intersected_by_lines)-len(set(boxes_intersected_by_lines))} boxes intersected by more than one line')
	if len(boxes_intersected_by_lines)-len(set(boxes_intersected_by_lines)) > 0:
		print(f'[W]: Resolving the "one box intersected by several lines" conflict by just adding the box to each object individually, i.e. resulting objects may overlap')
	new_boxes = [box for i,box in enumerate(all_boxes) if i in boxes_not_intersected_by_lines]
	for k in line_intersects.keys():
		if len(line_intersects[k]) > 0:
			new_boxes.append(all_boxes[line_intersects[k][0]])
		for i in line_intersects[k][1:]:
			new_boxes[-1] += all_boxes[i]
	if TRANSVERSED_LINES_OUTPUT_HULL:
		new_all_boxes = []
		for mesh in new_boxes:
			hull = return_convex_hull(mesh)
			box = return_box(mesh)
			try:
				hullV = hull.get_volume()
			except: pass
			try:
				boxV = box.get_volume()
			except: pass
			#print(f'BoxVolume/ConvexHullVolume = {boxV/hullV}')
			new_all_boxes.append(hull)
		all_boxes = new_all_boxes
	else:
		all_boxes = new_boxes

# *********************AD-HOC 'SMART' OPERATION**************************
# Join together all the boxes that are closer than DISTANCE meters
#************************************************************************
if JOIN_CLOSE_ONES:
	def are_close_enough(mesh1,mesh2):
		"""
		Return a boolean indicating if the meshes are 'close enough'
		"""
		DISTANCE = 0.3 # unit is meters
		return cdist(np.asarray(mesh1.vertices), np.asarray(mesh2.vertices), 'euclidean').flatten().min() < DISTANCE
	was_absorbed = []
	for i in range(len(all_boxes[:-1])):
		for j in range(i+1,len(all_boxes)):
			if j not in was_absorbed:
				if are_close_enough(all_boxes[i], all_boxes[j]):
					all_boxes[i] += all_boxes[j]
					was_absorbed.append(j)
	all_boxes = [return_convex_hull(box) for j,box in enumerate(all_boxes) if j not in was_absorbed]

# Twist back
for i in range(len(all_boxes)):
	all_boxes[i] = twist_back_meshObject(all_boxes[i],op)

# Record
mesh = o3d.geometry.TriangleMesh()
for v in all_boxes:
	mesh += v
o3d.io.write_triangle_mesh(f"triangleMesh.obj", mesh)

