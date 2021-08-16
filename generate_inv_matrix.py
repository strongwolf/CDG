import numpy as np 
import pickle
import os
theta = 0.01
lamb = 0.001

wbg = True
idl = True
bg_dict_num = 192
gt_dict_num = 192
feat_dim = 4096
dict_root_dir = ''
gt_dicts_file = os.path.join(dict_root_dir, 'gt_learn_dicts_' + str(gt_dict_num) + '_' + str(theta) + '.pkl')
bg_dicts_file = os.path.join(dict_root_dir, 'bg_learn_dicts_' + str(bg_dict_num) + '_' + str(theta) + '.pkl')
with open(gt_dicts_file, 'rb') as f:
	gt_dicts = pickle.load(f)
with open(bg_dicts_file, 'rb') as f:
	bg_dicts = pickle.load(f)
if idl:
	idl_inv_file = os.path.join(dict_root_dir, 'idl_learn_inv_'  + str(bg_dict_num) +'_' +  str(gt_dict_num) + '_'+ str(theta) + '_' + str(lamb) + '.pkl')
	num_classes = len(gt_dicts) + 1
	idl_invs = []
	for i in range(num_classes):
		if i == 0:
			dict = bg_dicts[0]
		else:
			dict = gt_dicts[i-1]
		t = np.dot(dict.T, dict) + lamb * np.eye(dict.shape[1])
		inv = np.dot(np.linalg.inv(t), dict.T)
		idl_invs.append(inv)
	with open(idl_inv_file, 'wb') as f:
		pickle.dump(idl_invs, f)

bg_dicts.extend(gt_dicts)
if wbg:
	dicts = bg_dicts
	inv_file = os.path.join(dict_root_dir, 'wbg_learn_inv_' + str(bg_dict_num) +'_' +  str(gt_dict_num) + str(theta) + '_' + str(lamb) +'.pkl')
else:
	dicts = gt_dicts
	inv_file = os.path.join(dict_root_dir, 'learn_inv_' + str(bg_dict_num) +'_' +  str(gt_dict_num)  + str(theta) + '_' + str(lamb) +'.pkl')
if type(dicts) == list:
	num_classes = len(dicts)
	dict_num = dicts[0].shape[1]
	#dicts  = np.stack(dicts, 1).reshape(feat_dim, -1)
	dicts  = np.concatenate(dicts, 1).reshape(feat_dim, -1)
t = np.dot(dicts.T, dicts) + lamb * np.eye(dicts.shape[1])

inv = np.dot(np.linalg.inv(t), dicts.T)

with open(inv_file, 'wb') as f:
	pickle.dump(inv, f)
if wbg:
	new_dicts_file = os.path.join(dict_root_dir, 'wbg_learn_dicts_'+ str(bg_dict_num) +'_' +  str(gt_dict_num) + str(theta) + '.pkl')
	with open(new_dicts_file, 'wb') as f:
		pickle.dump(dicts, f)

	