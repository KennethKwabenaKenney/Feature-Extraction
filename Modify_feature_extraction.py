# -*- coding: utf-8 -*-
"""
Last updated April/23/2023

@author: Kenneth Kwabena Kenney

"""

import laspy
import numpy as np
import tkinter as tk
from tkinter import filedialog
import open3d as o3d
from scipy.spatial import cKDTree
import cv2


#%% Functions
class structtype():
    pass

def readPtcloud(filePath, param):
    L = laspy.read(filePath)
    ptcloud = np.array((L.x,L.y,L.z,L.intensity,L.red,L.green,L.blue,L.gps_time,L.classification)).transpose()
    ptcloud = np.column_stack((ptcloud,np.arange(len(ptcloud)).astype(float))) # point ID (9)
    param.num_pts = len(ptcloud)
    return ptcloud, param.num_pts 

def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def voxel_downSampling(pc,voxelsize):
    rcd = np.zeros((pc.shape[0],10))
    minXYZ = np.min(pc[:,:3],axis=0)
    rcd[:,:3] = np.floor((pc[:,:3] - minXYZ)/voxelsize)
    
    # Assign color and classification fields to the appropriate columns
    rcd[:, 3:6] = pc[:, 3:6]  # Assuming columns 3-5 contain RGB color values
    rcd[:, 6:8] = pc[:, 6:8]  # Assuming columns 6-7 contain classification values
    rcd[:, 8:10] = pc[:, 8:10] # Assuming columns 8-10 contain additional information
    
    #idx = (np.max(rcd[:,0])*np.max(rcd[:,1]))*(rcd[:,2]) + np.max(rcd[:,0])*(rcd[:,1]) + rcd[:,0]   # n = max + 1
    nx = np.max(rcd[:,0]) + 1
    ny = np.max(rcd[:,1]) + 1
    nz = np.max(rcd[:,2]) + 1
    max_idx = nx * ny * nz
    idx = nx * ny * rcd[:,2] + nx * rcd[:,1] + rcd[:,0] 
    #idx -= np.min(idx) # too large index can lead to errors 
    pc = np.column_stack((pc,idx))
    rcd[:,9] = idx
    rcd = rcd.astype(int) # make it faster than float   
    #pc_ds = util.unique_rows(rcd) # much faster than np.unique    check this make it unique based on rcd[:, 9] idx
    
    uni_idx, idx_uni = np.unique(rcd[:,9], True)
    
    pc_ds = rcd[idx_uni]   
    pc_ds = pc_ds.astype(float)
    pc_ds[:,:3] = minXYZ + (pc_ds[:,:3] + voxelsize/2) * voxelsize    
    return pc, pc_ds

def plotPC(pc,color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,0:3])
    t = scaleData(color)
    t = -(t-1)
    t = np.uint8(255*t)
    t_c = cv2.applyColorMap(t, cv2.COLORMAP_JET)/255
    t_c = t_c[:,0,:]
    pcd.colors = o3d.utility.Vector3dVector(t_c)
    o3d.visualization.draw([pcd])
    
#%% Parameters & File Load

param = structtype()
param.fileEXtension = 'las'
param.check_unit = 0 # 1 is feet
param.radius = 5 
param.max_nn = 25
param.voxelsize = 0.1
radius = 0.5

root = tk.Tk()
filePath = filedialog.askopenfilename() 
root.withdraw()

#%% Read Ptc Array, Downsample & Below ground point filtering

# Read the point cloud & the number of ptcs
ptc, param.num_pts = readPtcloud(filePath, param)
# ptc, ptc_ds = voxel_downSampling(ptc, param.voxelsize) # downsampling 

# Building individual ptc arays
x = ptc[:, 0]
y = ptc[:, 1]
z = ptc[:, 2]
intensity = ptc[:, 3]
red = ptc[:, 4]
green = ptc[:, 5]
blue = ptc[:, 6]
gps_time = ptc[:, 7]
classification = ptc[:, 8]


# Find the indices where L.classification is 1 and 2
indices_class1 = np.where(classification == 1)[0]   # off ground points
indices_class2 = np.where(classification == 2)[0]   # ground points

### Average method of filtering points below ground level 
# Find the average L.z value for L.classification value 2
avg_z_class2 = np.mean(z[indices_class2])

# Filter out points with L.classification value 1 and L.z below min_z_class2
filtered_indices_class1 = indices_class1[z[indices_class1] > avg_z_class2]

#%% (Section 1) Use this to include ground points in the final output, comment Section 2

# #Combine filtered indices from class 1 and all indices from class 2
# filtered_indices = np.concatenate((filtered_indices_class1, indices_class2))

# # Convert the filtered indices list to a NumPy array
# filtered_indices = np.array(filtered_indices)

#%% Iterative method for ground point removal
## Iteration method of filtering points below ground level 
# # Iterate through each value of L.z for points with L.classification value 1
# for idx_class1 in indices_class1:
#     z_class1 = z[idx_class1]
    
#     # Check if there are any points with L.classification value 2 that have a higher L.z value
#     if np.any(z[indices_class2] > z_class1):
#         filtered_indices.append(idx_class1)

#%% (Second 2) Use this to exclude ground points in the final output, Comment Section 1

# Convert the filtered indices list to a NumPy array
filtered_indices = np.array(filtered_indices_class1)


#%% Continue with the process

# Create a filtered point cloud array
fltrd_ptc = np.array((x[filtered_indices], y[filtered_indices], z[filtered_indices], 
                             intensity[filtered_indices], red[filtered_indices], 
                             green[filtered_indices], blue[filtered_indices], 
                             gps_time[filtered_indices], classification[filtered_indices])).transpose()

num_fltrd_ptc = len(fltrd_ptc)
#%%  feature extraction (linearity, planarity, scattering)

xyz_ptc = fltrd_ptc[:,0:3]  # Extract the x, y & z values from the filtered ptc

# has_nans = np.isnan(xyz_ptc)
# if np.any(has_nans):
#     print("The array contains NaN values.")
# else:
#     print("The array does not contain NaN values.")

# has_inf = np.isinf(xyz_ptc)
# if np.any(has_inf):
#     print("The array contains Inf values.")
# else:
#     print("The array does not contain Inf values.")

# param.num_ds_pts = len(xyz_ptc)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz_ptc[:,0:3])
# pcd_tree = o3d.geometry.KDTreeFlann(pcd)
# features = []
# for i in range(xyz_ptc.shape[0]):
#     [__, idx, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[i], param.radius, param.max_nn)
        
#     covariance = np.cov(xyz_ptc[idx,0:3].T,)
#     eigenvalue, eigenvector = eig(covariance)
                
#     e0 = np.sqrt(np.clip(np.min(eigenvalue), 0, None))
#     e1 = np.sqrt(np.clip(np.median(eigenvalue), 0, None))
#     e2 = np.sqrt(np.clip(np.max(eigenvalue), 0, None))

#     linearity = (e2-e1)/e2
#     planarity = (e1-e0)/e2
#     scattering = (e0)/e2

#     features.append([linearity, planarity, scattering])
        
#     # if i % 1000 == 0:
#     #     print("Iteration: %d/%d" % (i, param.num_ds_pts)) 
        
######################################################################

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz_ptc[:,0:3])
# pcd_tree = o3d.geometry.KDTreeFlann(pcd)
# features = []
# for i in range(xyz_ptc.shape[0]):
#     [__, idx, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[i], param.radius, param.max_nn)
        
#     # # Extract neighboring points
#     # neighbors = xyz_ptc[idx, 0:3]
    
#     # # Compute the covariance matrix
#     # cov_matrix = np.cov(neighbors, rowvar=False)
    
#     # Calculate eigenvalues and eigenvectors of the covariance matrix
#     eigenvalues, eigenvectors = LA.eig(xyz_ptc[idx,0:3]) 
   
#     # eigenvalue, eigenvector = eig(np.diag((xyz_ptc[idx,0:3].T)))
    
#     e0 = np.sqrt(np.clip(np.min(eigenvalues), 0, None))
#     e1 = np.sqrt(np.clip(np.median(eigenvalues), 0, None))
#     e2 = np.sqrt(np.clip(np.max(eigenvalues), 0, None))

#     linearity = (e2-e1)/e2
#     planarity = (e1-e0)/e2
#     scattering = (e0)/e2


#############################################


# Construct KD-Tree
kdtree = cKDTree(xyz_ptc[:, 0:3])

# Initialize lists to store features
# linearity_values = []
# planarity_values = []
# scattering_values = []
features = []

#n_test = 10000

for i in range(xyz_ptc.shape[0]):
#for i in range(n_test):
    # Find neighbors within a specified radius
    neighbors = kdtree.query_ball_point(xyz_ptc[i, 0:3], radius)
    
    if len(neighbors) < 3:
        # Handle the case where there are not enough neighbors for the calculation
        # linearity_values.append(np.nan)
        # planarity_values.append(np.nan)
        # scattering_values.append(np.nan)
        features.append([np.nan, np.nan, np.nan])
        continue
    
    # Extract neighboring points
    neighbor_points = xyz_ptc[neighbors, 0:3]
    
    # Compute the covariance matrix
    cov_matrix = np.cov(neighbor_points, rowvar=False)
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(cov_matrix)
    
    # Sort eigenvalues
    eigenvalues.sort()
    
    # Calculate features
    e0, e1, e2 = eigenvalues
    linearity = (e2 - e1) / e2
    planarity = (e1 - e0) / e2
    scattering = (e0) / e2
    
    # Store the features
    # linearity_values.append(linearity)
    # planarity_values.append(planarity)
    # scattering_values.append(scattering)
    features.append([linearity, planarity, scattering]) 

    if i % 100000 == 0:   
        print("Iteration: %d/%d" % (i, num_fltrd_ptc)) 
        
        
features = np.array(features)
#classification_ds = np.argmax(features, axis=1) 
#plotPC(fltrd_ptc,classification_ds)  # 0: linearity (blue), 1: planarity (green), 2: scattering (red)



#plotPC(fltrd_ptc,features)

pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(fltrd_ptc[0:n_test,0:3])
#pcd.colors = o3d.utility.Vector3dVector(features[0:n_test,0:3])
pcd.points = o3d.utility.Vector3dVector(fltrd_ptc[:,0:3])
pcd.colors = o3d.utility.Vector3dVector(features[:,0:3])
o3d.visualization.draw([pcd])






#%% Save Output/Result for Visualization

# root = tk.Tk()
# output_file_path = filedialog.asksaveasfilename(defaultextension=".laz", filetypes=[("LAZ Files", "*.laz"), ("LAS Files", "*.las")])
# root.withdraw()
    
# # if output_file_path:
#     # Create a new LAS file
#     out_las = laspy.create(file_version="1.4", point_format=7)
#     out_las.header.offset = [np.min(ptc[:, 0]), np.min(ptc[:, 1]), np.min(ptc[:, 2])]
#     out_las.header.scale = [0.1, 0.1, 0.1]
#     # Add point attributes
#     out_las.x = fltrd_ptc[:, 0]
#     out_las.y = fltrd_ptc[:, 1]
#     out_las.z = fltrd_ptc[:, 2]
#     out_las.intensity = fltrd_ptc[:, 3]
#     out_las.red = fltrd_ptc[:, 4]
#     out_las.green = fltrd_ptc[:, 5]
#     out_las.blue = fltrd_ptc[:, 6]
#     out_las.gps_time = fltrd_ptc[:, 7]
#     out_las.classification = fltrd_ptc[:, 8]
    
#     # Save the LAS file
#     out_las.write(output_file_path)
    
