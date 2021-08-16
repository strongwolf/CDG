import numpy as np 
#from cvxopt import matrix
import pickle
from l1ls import l1ls
import copy
import pdb
import os

feats_file = 'gt_feats.pkl'
mode = 'gt'
with open(feats_file, 'rb') as f:
    feats = pickle.load(f)   
num_classes = len(feats)
dicts_list = []
dicts_num = 192
max_iters = 25
min_tol = 1e-2
lamda = 1e-3
learn_dicts_list = []
learn_alpha_list = []
error_pre = 0
error_now = 0
error_list = []
for i in range(num_classes):
    error = []
    feat = feats[i]
    init_dict = np.random.randn(feat.shape[0], dicts_num)
    learn_dict = None
    norm = np.linalg.norm(init_dict, axis=0, keepdims=True)
    init_dict = init_dict / norm
    print('Begin learn class {} \n'.format(i))
    num_sample = feat.shape[1]
    for k in range(max_iters):
        alpha = []
        if k == 0:
            dict = init_dict
        else:
            dict = learn_dict
        for j in range(feat.shape[1]):
            [x, status, hist] = l1ls(dict, feat[:,j], lamda, quiet=True)
            if 'Failed' in status:
                print('L1 normalization not solved!')
            alpha.append(x.reshape(-1,1))
        alpha = np.concatenate(alpha, axis=1)
        recon_feat = np.matmul(dict, alpha)
        learn_dict = []
        for j in range(dict.shape[1]):
            y = feat - (recon_feat - dict[:,[j]].reshape(-1,1) * alpha[[j],:].reshape(1,-1))
            d_j = np.matmul(y, alpha[j, :].reshape(-1, 1))
            norm_d_j = d_j / np.linalg.norm(d_j)
            learn_dict.append(norm_d_j.reshape(-1, 1))
        learn_dict = np.concatenate(learn_dict, axis=1)
        recon_error = ((feat - np.matmul(learn_dict, alpha))**2).sum() / num_sample
        co_error = np.abs(alpha).sum() * lamda / num_sample
        error.append([recon_error, co_error])
        error_pre = error_now
        error_now = recon_error + co_error
        print('iter: {}  error: {} {} \n'.format(k, recon_error, co_error))
        if abs(error_now - error_pre) < min_tol:
            break
    learn_dicts_list.append(learn_dict)
    learn_alpha_list.append(alpha)
    error_list.append(error)

dict_file = os.path.join(os.path.dirname(feats_file), mode + '_learn_dicts_'+ str(lamda) +'.pkl')
alpha_file = os.path.join(os.path.dirname(feats_file), mode +'_alpha_' + str(lamda) +'.pkl')
error_file =  os.path.join(os.path.dirname(feats_file), mode +'_error_' + str(lamda) +'.pkl')
with open(dict_file, 'wb') as f:
    pickle.dump(learn_dicts_list, f)
with open(alpha_file, 'wb') as f:
    pickle.dump(learn_alpha_list, f)
with open(error_file, 'wb') as f:
    pickle.dump(error_list, f)