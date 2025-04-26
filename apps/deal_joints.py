import json

import numpy as np
#
# [0,69,6,8,41,5,7,62,-1,10,12,14,9,11,13]
def sapiens_joints_tokenhmr(json_path,image_height=224,image_width=224):
    ids=np.array([0,69,6,8,41,5,7,62,-1,10,12,14,9,11,13,2,1,4,3,15,16,17,18,19,20], dtype=np.int32)

    # keypoints_np=np.zeros((15,2))
    with open(json_path, 'r') as f:
        keypoints=json.load(f)
        keypoints_np=np.array(keypoints["instance_info"][0]['keypoints'])
        keypoints_np=keypoints_np[ids]
        keypoints_np[ids < 0] *= 0.0  # remove undefined keypoints
        # keypoints_np[8]=(keypoints_np[9]+keypoints_np[12])/2
        for i in range(25):
            keypoints_np[i][0]=(keypoints_np[i][0]/(image_width/2))-1
            keypoints_np[i][1]=(keypoints_np[i][1]/(image_height/2))-1

    return keypoints_np

def sapiens_joints_gcmr(json_path,image_height=224,image_width=224):
    ids=np.array([16,14,12,11,13,15,10,8,6,5,7,9,0,1,2,3,4], dtype=np.int32)

    # keypoints_np=np.zeros((15,2))
    with open(json_path, 'r') as f:
        keypoints=json.load(f)
        keypoints_np=np.array(keypoints["instance_info"][0]['keypoints'])
        keypoints_np=keypoints_np[ids]
        keypoints_np[ids < 0] *= 0.0  # remove undefined keypoints
        # keypoints_np[8]=(keypoints_np[9]+keypoints_np[12])/2
        for i in range(17):
            keypoints_np[i][0]=(keypoints_np[i][0]/(image_width/2))-1
            keypoints_np[i][1]=(keypoints_np[i][1]/(image_height/2))-1

    return keypoints_np

