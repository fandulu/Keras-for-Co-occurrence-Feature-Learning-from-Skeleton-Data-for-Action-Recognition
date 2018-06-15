import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import random
import os
import glob
import scipy.ndimage.interpolation as inter

class SBU_dataset():
    def __init__(self, dir):
        print ('loading data from:', dir)
        
        self.pose_paths = glob.glob(os.path.join(dir, 's*', '*','*','*.txt'))
        self.pose_paths.sort()
        
    
    def get_data(self, test_set_folder):
               
        cross_set = {}
        cross_set[0] = ['s01s02', 's03s04', 's05s02', 's06s04']
        cross_set[1] = ['ds02s03', 's02s07', 's03s05', 's05s03']
        cross_set[2] = ['s01s03', 's01s07', 's07s01', 's07s03']
        cross_set[3] = ['s02s01', 's02s06', 's03s02', 's03s06']
        cross_set[4] = ['s04s02', 's04s03', 's04s06', 's06s02', 's06s03']
        
        def read_txt(pose_path):
            a = pd.read_csv(pose_path,header=None).T
            a = a[1:]
            return a.as_matrix()
        
        print('test set folder should be slected from 0 ~ 4')
        print('slected test folder {} includes:'.format(test_set_folder), cross_set[test_set_folder])

        train_set = []
        test_set = []
        for i in range(len(cross_set)):
            if i == test_set_folder:
                test_set += cross_set[i]
            else:
                train_set += cross_set[i]

        train = {} 
        test = {}

        for i in range(1,9):
            train[i] = []
            test[i] = []

        for pose_path in self.pose_paths:
            pose = read_txt(pose_path)
            if pose_path.split('/')[-4] in train_set:   
                train[int(pose_path.split('/')[-3])].append(pose) 
            else:
                test[int(pose_path.split('/')[-3])].append(pose) 

        return train, test
        

#Transfer to orginial coordinates for plotting
def coord2org(p): 
    p_new = np.empty_like(p)
    for i in range(15):
        p_new[i,0] = 640 - (p[i,0] * 640)
        p_new[i,1] = 480 - (p[i,1] * 240)
    return p_new

#Plotting the pose
def draw_2d_pose(gtorigs): 
    f_ind = np.array([
        [2,1,0],
        [3,6,2,3],
        [3,4,5],
        [6,7,8],
        [2,12,13,14],
        [2,9,10,11],      
    ])

    fig = plt.figure()
    
    axes = plt.gca()
    axes.set_xlim([0,640])
    axes.set_ylim([0,480])

    ax = fig.add_subplot(111)
    
    for gtorig,color in zip(gtorigs,['r','b']):
        
        gtorig = coord2org(gtorig)
        
        for i in range(f_ind.shape[0]):
        
            ax.plot(gtorig[f_ind[i], 0], gtorig[f_ind[i], 1], c=color)
            ax.scatter(gtorig[f_ind[i], 0], gtorig[f_ind[i], 1],s=10,c=color)
        
    plt.show()

#Rescale to be 16 frames
def zoom(p):
    l = p.shape[0]
    p_new = np.empty([16,15,3]) 
    for m in range(15):
        for n in range(3):
            p_new[:,m,n] = inter.zoom(p[:,m,n],16/l)[:16]
    return p_new

#Switch two persons' position
def mirror(p_0,p_1):
    p_0_new = np.copy(p_0)
    p_1_new = np.copy(p_1)
    p_0_new[:,:,0] = abs(p_0_new[:,:,0]-1)
    p_1_new[:,:,0] = abs(p_1_new[:,:,0]-1)
    return p_0_new, p_1_new
