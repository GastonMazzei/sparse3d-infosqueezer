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


# Define a function that adds walls
def add_outer_walls(v):
	"""
	PEND: Make Function That Makes Walls
	"""
	return v			

# Open file
if False:
	df = pd.read_csv('raw_data/pointclouds.csv')
	xyz = df.loc[df.groupby(['id'])['q'].idxmax()]
	HARDCODED_THRESHOLD = 0.001
	results = xyz[xyz['q'] > HARDCODED_THRESHOLD][['x','z','y']].to_numpy().astype('float')
	ptcloud = results.copy()
	#ptcloud[:,0] = results[:,2].copy()
	#ptcloud[:,2] = results[:,0].copy()
	ptcloud[:,2] *= -1

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(ptcloud)
else:
	input_filename = 'raw_data/original_pcd_aligned.ply'
	pcd = o3d.io.read_point_cloud(input_filename) 



# Try to write data as "x y z" in a file
if 'x_y_z_for_line_segmentation.txt' not in os.listdir('raw_data'):
	with open('raw_data/x_y_z_for_line_segmentation.txt','w') as file:
		for v in np.asarray(pcd.points):
			file.write(f'{v[0]} {v[1]} {v[2]}\n')


# Useless plane detector
def plane_segmenter(pcd, draw=False):
	plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
		                                 ransac_n=100,
		                                 num_iterations=3000)
	[a, b, c, d] = plane_model
	print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
	inlier_cloud = pcd.select_by_index(inliers)
	inlier_cloud.paint_uniform_color([0.5, 0, 0])
	outlier_cloud = pcd.select_by_index(inliers, invert=True)
	outlier_cloud.paint_uniform_color([0.5, 0.5, 0])
	if draw: o3d.visualization.draw_geometries([inlier_cloud])#, outlier_cloud])
	return np.asarray(inlier_cloud.points),outlier_cloud,(a,b,c,d)
planes = []
excluded = [] 
for _ in range(100):
	backup_pcd = np.asarray(pcd.points)
	np.random.shuffle(backup_pcd)
	try:
	#if True:
		temp, pcd, planeParams = plane_segmenter(pcd, False)
		# Filter by number of points
		if len(np.asarray(pcd.points)) < 30: continue
		# Filter by direction of the plane
		if abs(planeParams[1]) > 0.95 and False: 
			pcd.points = o3d.utility.Vector3dVector(backup_pcd)
			continue
		planes.append(temp.copy())
	except: break
	#else: break
try:
	os.system(f'rm -r planes')
except: pass
try:
	os.mkdir(f'planes')
except: pass
c=0
for p in planes:
	pcd_out = o3d.geometry.PointCloud()
	pcd_out.points = o3d.utility.Vector3dVector(p)
	o3d.io.write_point_cloud(f"planes/segmented_plane{c}.ply", pcd_out)
	c+=1





# Exit prematurely, testing.
sys.exit(1)

def filter_labels(labels, MIN_ACCEPTED_N=30):
	# Count cluster appearances and set to noise those that dont have enough points
	d = dict([(i,0) for i in np.unique(labels)])
	for number in labels:
	    d[number]+=1
	labels = [l if d[l]>MIN_ACCEPTED_N else -1 for l in labels]
	return np.asarray(labels)


MIN_ACCEPTED_NUMBER_OF_POINTS = 30

# EXPLORING
EXPLORING = [False,True][1]
if EXPLORING:
	Nclusters = []
	ALL_EPS = np.linspace(0.01,0.8,30)
	ALL_MIN_POINTS = [2 + 4 * j for j in range(20)]
	DISPLAY = False
	for EPS in ALL_EPS:
		for min_points in ALL_MIN_POINTS:
			labels = np.array(
				pcd.cluster_dbscan(eps=EPS, min_points=min_points, print_progress=True))
			max_label = labels.max()

		
			labels = filter_labels(labels, MIN_ACCEPTED_NUMBER_OF_POINTS)

			print(f"for eps = {EPS} and min_points = {min_points} point cloud has {len(set(labels))+1} clusters")
			if DISPLAY:
				colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
				colors[labels < 0] = 0
				pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
				o3d.visualization.draw_geometries([pcd])
			Nclusters.append(len(set(labels))+1)
	result = np.asarray(Nclusters).reshape(len(ALL_EPS), len(ALL_MIN_POINTS))
	plt.imshow(result); plt.colorbar(); plt.show()
	target_i, target_j = [int(x) for x in input('Please write the comma-separated ixes, starting w the vertical dimension. E.g. "11,5"').split(',')]
	print(f'Target ixes are {target_i},{target_j}')


# COMMITTING TO ONE VALUE
#eps = 0.3 and min_points = 10 40 clusters
# eps = 0.3 and min_points = 25 point 20 clusters
#eps = 0.3 and min_points = 1 point cloud has 156 clusters
#for eps = 0.5 and min_points = 50 point cloud has 11 clusters
#eps = 0.1 and min_points = 1
#eps = 0.2 and min_points = 7 104 clusters
def search(ALL_EPS,ALL_MIN_POINTS,result,target_i=11, target_j=5):
	HARDCODED_IXES = [False, True][1]
	for i,_EPS in enumerate(ALL_EPS):
		for j,_min_points in enumerate(ALL_MIN_POINTS):
			# Look for specific ixes
			if HARDCODED_IXES:
				if i==target_i and j==target_j: return _EPS,_min_points
			else:
				# Look for the min
				if result[i,j] == np.min(result):
					print(f'chosen ixes are {i} and {j}. Min is {np.min(result)}')
					return _EPS, _min_points

EPS, min_points = search(ALL_EPS, ALL_MIN_POINTS, result, target_i, target_j)
labels = np.array(pcd.cluster_dbscan(eps=EPS, min_points=min_points, print_progress=True))
labels = filter_labels(labels, MIN_ACCEPTED_NUMBER_OF_POINTS)
max_label = labels.max()


try:
	os.system(f'rm -r segmented')
except: pass


try:
	os.mkdir(f'segmented')
except: pass

print(list(set(labels)))
# Save the point clouds per room :-)
c=0
for i in range(max_label + 1):
	sub_ptcloud = np.asarray(pcd.points)[labels == i]
	if len(sub_ptcloud)==0: continue
	pcd_out = o3d.geometry.PointCloud()
	pcd_out.points = o3d.utility.Vector3dVector(sub_ptcloud)
	o3d.io.write_point_cloud(f"segmented/segmented_object{c}.ply", pcd_out)
	c+=1









