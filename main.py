import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2
import copy

CALIBRATION_IMGS_PATH = "calibration_imgs/"

SQUARES_X, SQUARES_Y = 9, 6
SQUARE_LENGTH = 21.5
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def load_imgs(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(folder_path + filename)
        images.append(img)
    return images

def get_corners(img):
    img_edit = copy.deepcopy(img)
    img_edit = cv2.cvtColor(img_edit,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img_edit,(SQUARES_X,SQUARES_Y),None)
    if ret:
        #img_draw = cv2.drawChessboardCorners(copy.deepcopy(img),(SQUARES_X,SQUARES_Y),corners,ret)
        #plt.imshow(img_draw)
        #plt.show()
        corners_subpix = cv2.cornerSubPix(img_edit,corners,(7,7),(-1,-1),SUBPIX_CRITERIA)
        corners_subpix = corners_subpix.reshape((-1,2))
        #print(corners_subpix)
        return corners_subpix
    return None

def calculate_homography(img1_sample,img2_sample):
    A = []
    img1_sample = img1_sample.reshape(-1,1,2)
    img2_sample = img2_sample.reshape(-1,1,2)
    for i in range(img1_sample.shape[0]):
        x1, y1 = img1_sample[i][0][0], img1_sample[i][0][1]
        x2, y2 = img2_sample[i][0][0], img2_sample[i][0][1]
        A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
        A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])
    A = np.array(A).astype(np.float32)
    U, S, V = np.linalg.svd(A,full_matrices=True)
    homography = V[8,:].reshape((3, 3))
    homography = homography/homography[2,2]
    return homography

def single_v_vector(hi,hj):
    return np.array([hi[0]*hj[0], hi[0]*hj[1] + hi[1]*hj[0], hi[1]*hj[1], hi[2]*hj[0] + hi[0]*hj[2], hi[2]*hj[1] + hi[1]*hj[2], hi[2]*hj[2]]).T

def combined_v_vector(homographies):
    V = []
    for H in homographies:
        h1, h2 = H[:,0], H[:,1]
        V11 = single_v_vector(h1,h1)
        V12 = single_v_vector(h1,h2)
        V22 = single_v_vector(h2,h2)
        V.append(V12.T)
        V.append((V11-V22).T)
    return np.array(V)

def B_matrix(combined_v_vector):
    B = np.zeros((3,3))
    U, S, V = np.linalg.svd(combined_v_vector)
    b = V[-1,:]
    B[0,0] = b[0]
    B[0,1] = b[1]
    B[1,0] = b[1]
    B[1,1] = b[2]
    B[2,0] = b[3]
    B[0,2] = b[3]
    B[1,2] = b[4]
    B[2,1] = b[4]
    B[2,2] = b[5]
    #print(B[0,0])
    return B

def intrinsic_matrix(B):
    c_y = (B[0,1]*B[0,2] - B[0,0]*B[1,2])/(B[0,0]*B[1,1] - B[0,1]**2)
    lamb = B[2,2] - (B[0,2]**2 + c_y*(B[0,1]*B[0,2] - B[0,0]*B[1,2]))/B[0,0]
    #print(B[0,0])
    f_x = np.sqrt(lamb/B[0,0])
    f_y = np.sqrt(lamb*(B[0,0]/(B[0,0]*B[1,1] - B[0,1]**2)))
    gamma = -(B[0,1]*(f_x**2)*f_y)/lamb
    c_x = (gamma*c_y/f_y) - (B[0,2]*(f_x**2)/lamb)
    return np.array([[f_x,gamma,c_x],[0,f_y,c_y],[0,0,1]]).astype(np.float32)

def calc_rot_trans_matrices(K,homographies):
    matrices = []
    for H in homographies:
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]
        lamb = np.linalg.norm(np.dot(np.linalg.inv(K),h1),2)
        r1 = np.dot(np.linalg.inv(K),h1)/lamb
        r2 = np.dot(np.linalg.inv(K),h2)/lamb
        r3 = np.cross(r1,r2)
        t = np.dot(np.linalg.inv(K),h3)/lamb
        matrix = np.vstack((r1,r2,r3,t)).T
        matrix = np.concatenate((matrix,np.array([[0,0,0,1]])),axis=0).astype(np.float32)
        matrices.append(matrix)
    return matrices

def reprojection_error_single_image(intrinsic_params,base_corners,img_corners,orig_rot_trans_matrix):
    f_x, f_y, c_x, c_y, gamma, k1, k2 = list(intrinsic_params)
    K = np.array([[f_x,gamma,c_x],[0,f_y,c_y],[0,0,1]]).astype(np.float32)
    rot_trans_matrix = np.array([orig_rot_trans_matrix[:3,0],orig_rot_trans_matrix[:3,1],orig_rot_trans_matrix[:3,3]]).reshape(3,3)
    ext_int_matrix = np.dot(K,rot_trans_matrix)
    reprojection_error = 0
    for i in range(img_corners.shape[0]):
        base_corner_3D = np.array([base_corners[i,0],base_corners[i,1],0,1]).reshape(4,1)
        base_corner_2D = np.array([base_corners[i,0],base_corners[i,1],1]).reshape(3,1)
        img_corner_3D = np.array([img_corners[i,0],img_corners[i,1],0,1]).reshape(4,1)
        img_plane_proj_corner = np.dot(orig_rot_trans_matrix,base_corner_3D).reshape(4)
        img_plane_proj_corner = img_plane_proj_corner/img_plane_proj_corner[2]
        x, y = img_plane_proj_corner[0], img_plane_proj_corner[1]
        distortion_radius = ((x)**2 + (y)**2)**0.5
        sensor_plane_proj_corner = np.dot(ext_int_matrix,base_corner_2D).reshape(3)
        sensor_plane_proj_corner = sensor_plane_proj_corner/sensor_plane_proj_corner[2]
        u, v = sensor_plane_proj_corner[0], sensor_plane_proj_corner[1]
        u_dash = c_x + (u-c_x)*(k1*(distortion_radius**2) + k2*(distortion_radius**4))
        v_dash = c_y + (v-c_y)*(k1*(distortion_radius**2) + k2*(distortion_radius**4))
        img_corner_3D_proj = np.array([u_dash,v_dash,0,1]).reshape(4)
        error = np.linalg.norm(img_corner_3D_proj-img_corner_3D)
        reprojection_error += error
    return reprojection_error
    
def total_reprojection_error(intrinsic_params,base_corners,all_img_corners,rot_trans_matrices):
    reprojection_errors = []
    for i in range(len(all_img_corners)):
        reprojection_errors.append(reprojection_error_single_image(intrinsic_params,base_corners,all_img_corners[i],rot_trans_matrices[i]))
    return np.array(reprojection_errors)

def optimize_distortion_params(old_intrinsic_params,base_corners,all_img_corners,rot_trans_matrices):
    ls_params = least_squares(fun=total_reprojection_error,x0=old_intrinsic_params,method="lm",args=[base_corners,all_img_corners,rot_trans_matrices])
    new_intrinsic_params = ls_params.x
    f_x, f_y, c_x, c_y, gamma, k1, k2 = list(new_intrinsic_params)
    K = np.array([[f_x,gamma,c_x],[0,f_y,c_y],[0,0,1]]).astype(np.float32)
    Kc = np.array([[k1],[k2]]).astype(np.float32)
    return K, Kc

def display_homography(rgb_img1,rgb_img2,H):
      # Both images must be of the same sizes to concatenate them side by side
      # H = np.linalg.inv(H)
      h_img = cv2.warpPerspective(copy.deepcopy(rgb_img1),H,(rgb_img2.shape[1],rgb_img2.shape[0]))
      img_concat = np.concatenate([rgb_img2.copy(),h_img],axis=1)
      plt.imshow(img_concat)
      plt.show()

def calibrate_wrapper(base_img,imgs):
    homographies = []
    base_corners = get_corners(copy.deepcopy(base_img))
    all_img_corners = []
    for img in imgs:
        img_corners = get_corners(copy.deepcopy(img))
        all_img_corners.append(img_corners)
        H = calculate_homography(base_corners,img_corners)
        #display_homography(img,base_img,H)
        #print(H)
        homographies.append(H)
    V = combined_v_vector(homographies)
    B = B_matrix(V)
    K = intrinsic_matrix(B)
    rot_trans_matrices = calc_rot_trans_matrices(K,homographies)
    old_instrinsic_params = np.array([K[0,0],K[1,1],K[0,2],K[1,2],K[0,1],0,0]).astype(np.float32)
    K, Kc = optimize_distortion_params(old_instrinsic_params,base_corners,all_img_corners,rot_trans_matrices)
    print("Camera Intrinsic Matrix (K):")
    print(K)
    print("*************************************************")
    print("Camera Distortion Matrix (Kc):")
    print(Kc)
    return K, Kc

def main():
    base_img = cv2.imread("checkerboard_pattern.jpg")
    imgs = load_imgs(CALIBRATION_IMGS_PATH)
    #img1 = cv2.imread("calibration_imgs/IMG_20170209_042606.jpg")
    #plt.imshow(img1)
    #plt.show()
    #get_corners(img1)
    K, rot_trans_matrix_list = calibrate_wrapper(base_img,imgs)
    '''print(K)
    print("*****************")
    print(rot_trans_matrix_list[0])'''

main()

'''if __name__ == "__main__":
    main()'''