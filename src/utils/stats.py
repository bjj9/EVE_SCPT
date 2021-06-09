import os
import math
import numpy as np

def getAllFilenames(directory, endsWith='', startsWith='', containsStr='', containsNot=None, returnWithFullDir=False, dirOnly=False):
    ''' from Bj utilities '''
    fileList = []
    for file in os.listdir(directory):
        if file.endswith(str(endsWith)):
            if file.startswith(str(startsWith)):
                if containsStr in file:
                    ta = file if not returnWithFullDir else os.path.join(directory, file)
                    if (containsNot is None) or (containsNot not in file):
                        if (not dirOnly) or (os.path.isdir(os.path.join(directory, file))):
                            fileList.append(ta)
    return fileList

def getAllSubjectFolder(main_dir, split=''):
    assert split in ['', 'val', 'train', 'test']
    # main_dir = 'E:/datasets/eve_dataset'
    # main_dir = 'E://codespace//pythonWorkingSpace//Bji//outputs//inference_output_refine_net_inference_30Hz'
    sample_paths = {}
    validataion_subjects = []
    all_split = ['val', 'train', 'test'] if split == '' else [split,]
    for ss in all_split:
        validataion_subjects = validataion_subjects + getAllFilenames(main_dir, startsWith=ss, returnWithFullDir=False)
    for sub in validataion_subjects:
        sub_dir = main_dir + '/' + sub
        sample_dirs = getAllFilenames(sub_dir, startsWith='step', containsNot='eye_tracker_calibration', returnWithFullDir=True, dirOnly=True)
        sample_paths[sub] = sample_dirs
    print('number of samples:', [(k, len(v)) for k, v in sample_paths.items()], '---refweddd')
    return sample_paths

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

## metrics
def metric_euc_initXY(D_ta, to_angle=False):
    return metric_euc(D_ta, ('gt_x', 'gt_y', 'hat_x_init', 'hat_y_init'), to_angle)
def metric_euc(D_ta, cols_to_compare, to_angle=False):
    ''' example: cols_to_compare = ('gt_x', 'gt_y', 'hat_x_init', 'hat_y_init')'''
    gt_x, gt_y, hat_x, hat_y = cols_to_compare
    errs = {}
    for subject in D_ta.keys():
        eee = D_ta[subject]
        err = metric_euc_for_one_subject(eee[gt_x], eee[gt_y], eee[hat_x], eee[hat_y])
        if to_angle:
            err /= 38
        errs[subject] = round(err,4)
    avg_err = round(np.mean([v for k, v in errs.items()]),4)
    print('errs for each subject', errs, '---wiefj')
    print('overall err (averaged by subject)', avg_err, '---nejskkk')
    return avg_err, errs
def metric_euc_for_one_subject(gt_x, gt_y, hat_x, hat_y):
    gt_x, gt_y, hat_x, hat_y = np.array(gt_x), np.array(gt_y), np.array(hat_x), np.array(hat_y)
    euc_error = np.sqrt((hat_x - gt_x)**2 + (hat_y - gt_y)**2)
    return np.mean(euc_error)




# make Gaussian masks
# def makeGaussianMasks()