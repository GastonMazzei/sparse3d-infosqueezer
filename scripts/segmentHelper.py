from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d




def Rotx(angle):
	return np.asarray([[1,0,0],
			[0,np.cos(angle),-np.sin(angle)],
			[0,np.sin(angle),np.cos(angle)]])


def Roty(angle):
	return np.asarray([[np.cos(angle),0,np.sin(angle)],
			[0,1,0],
			[-np.sin(angle),0,np.cos(angle)]])

def Rotz(angle):
	return np.asarray([[np.cos(angle),-np.sin(angle),0],
			[np.sin(angle),np.cos(angle),0],
			[0,0,1]])



# Get the pcd's points in the straight version
def straighten_pcd(pcd):
	with open(f'raw_data/boxwallsNormals.txt','r') as f:
		boxNormals = f.readlines()[0].split(',')
	points = np.asarray(pcd.points)
	boxNormals = [float(x) for x in boxNormals]
	assert(len(boxNormals) == 3)
	xval = 1 if boxNormals[0]>0 else -1
	mRoty = Roty(np.arccos(np.dot(np.asarray([xval,0,0]),np.asarray(boxNormals))))
	points = (mRoty @ points.T).T
	pcd_out = o3d.geometry.PointCloud()
	pcd_out.points = o3d.utility.Vector3dVector(points)
	return pcd_out, mRoty

def straighten_meshObj_fromOp(pcd,op=False):
	if type(op)==bool and op==False:
		with open(f'raw_data/boxwallsNormals.txt','r') as f:
			boxNormals = f.readlines()[0].split(',')
		boxNormals = [float(x) for x in boxNormals]
		assert(len(boxNormals) == 3)
		xval = 1 if boxNormals[0]>0 else -1
		op = Roty(np.arccos(np.dot(np.asarray([xval,0,0]),np.asarray(boxNormals))))
	try:
		points = np.asarray(pcd.points)
	except:
		points = np.asarray(pcd.vertices)
	points = (op @ points.T).T
	try:
		pcd.vertices = o3d.utility.Vector3dVector(points)
	except:
		pcd.points = o3d.utility.Vector3dVector(points)
	return pcd, op

def twist_back(points, operator):
	return (operator.T @ points.T).T

def twist_back_meshObject(meshObject, operator):
	try:
		points = np.asarray(meshObject.points)
	except:
		points = np.asarray(meshObject.vertices)
	new_points =  (operator.T @ points.T).T
	try:
		meshObject.points = o3d.utility.Vector3dVector(new_points)
	except:
		meshObject.vertices = o3d.utility.Vector3dVector(new_points)
	return meshObject














