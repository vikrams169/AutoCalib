# Loading the required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2
import copy

# Path to the folder with the images 
CALIBRATION_IMGS_PATH = "calibration_imgs/"
# Number of squares along each axis of the chessboard (exluding border squares)
SQUARES_X, SQUARES_Y = 9, 6
# Length of the square in mm (though not needed for this approach as digital chessboard corners are detected for the ground truth)
SQUARE_LENGTH = 21.5
# Criteria for getting sub-pixel level accurate corner detection
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Loading all the images of the chessboard at different angles from the specified directory path
def load_imgs(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(folder_path + filename)
        images.append(img)
    return images

# Getting the interior (not on the border) chessboard corners with sub-pixel level location accuracy for an image
def get_corners(img):
    # Creating a deepcopy of the image
    img_edit = copy.deepcopy(img)
    # Converting the image to a grayscale version
    img_edit = cv2.cvtColor(img_edit,cv2.COLOR_BGR2GRAY)
    # Detecting the chessboard corners
    ret, corners = cv2.findChessboardCorners(img_edit,(SQUARES_X,SQUARES_Y),None)
    if ret:
        # Detecting the sublixel accurate accurate corners
        corners_subpix = cv2.cornerSubPix(img_edit,corners,(7,7),(-1,-1),SUBPIX_CRITERIA)
        corners_subpix = corners_subpix.reshape((-1,2))
        return corners_subpix
    return None

# Calculating the approximate homography from paper coordinate system --> camera coordinate system using image and ground truth chessboard corners
def calculate_homography(img1_sample,img2_sample):
    # Initializing the A matrix
    A = []
    # Adding values from each corner association to the A matrix
    img1_sample = img1_sample.reshape(-1,1,2)
    img2_sample = img2_sample.reshape(-1,1,2)
    for i in range(img1_sample.shape[0]):
        x1, y1 = img1_sample[i][0][0], img1_sample[i][0][1]
        x2, y2 = img2_sample[i][0][0], img2_sample[i][0][1]
        A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
        A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])
    # Using singular value decomposition (SVD) to get the approximate homography
    A = np.array(A).astype(np.float32)
    U, S, V = np.linalg.svd(A,full_matrices=True)
    homography = V[8,:].reshape((3, 3))
    homography = homography/homography[2,2]
    return homography

# Generating a single V vector used in Zhang's Camera Calibration Algorithm
def single_v_vector(hi,hj):
    return np.array([hi[0]*hj[0], hi[0]*hj[1] + hi[1]*hj[0], hi[1]*hj[1], hi[2]*hj[0] + hi[0]*hj[2], hi[2]*hj[1] + hi[1]*hj[2], hi[2]*hj[2]]).T

# Constructing the V matrix from all individual V vectors from each image/homography as specified by Zhang's Camera Calibration Algorithm
def combined_v_vector(homographies):
    # Initializing the V vector
    V = []
    # Looping through all the images/homographies to construct the V matrix
    for H in homographies:
        h1, h2 = H[:,0], H[:,1]
        V11 = single_v_vector(h1,h1)
        V12 = single_v_vector(h1,h2)
        V22 = single_v_vector(h2,h2)
        V.append(V12.T)
        V.append((V11-V22).T)
    return np.array(V)

# Generating the B matrix (B = np.dot((K^-1).T),(K^-1) where K is the camera intrinsic matrix)
def B_matrix(combined_v_vector):
    # Initializing the B matrix and using Singular Value Decomposition (SVD) to get the individual entries of B
    B = np.zeros((3,3))
    U, S, V = np.linalg.svd(combined_v_vector)
    b = V[-1,:]
    # Reconstructing B from the individual entries
    B[0,0] = b[0]
    B[0,1] = b[1]
    B[1,0] = b[1]
    B[1,1] = b[2]
    B[2,0] = b[3]
    B[0,2] = b[3]
    B[1,2] = b[4]
    B[2,1] = b[4]
    B[2,2] = b[5]
    return B

# Getting intrinisc parameter values from the B matrix
def intrinsic_matrix(B):
    # Width pixel offset
    c_y = (B[0,1]*B[0,2] - B[0,0]*B[1,2])/(B[0,0]*B[1,1] - B[0,1]**2)
    lamb = B[2,2] - (B[0,2]**2 + c_y*(B[0,1]*B[0,2] - B[0,0]*B[1,2]))/B[0,0]
    # Focal length along the height
    f_x = np.sqrt(lamb/B[0,0])
    # Focal length along the width
    f_y = np.sqrt(lamb*(B[0,0]/(B[0,0]*B[1,1] - B[0,1]**2)))
    # Axis skew
    gamma = -(B[0,1]*(f_x**2)*f_y)/lamb
    # Height pixel offset
    c_x = (gamma*c_y/f_y) - (B[0,2]*(f_x**2)/lamb)
    return np.array([[f_x,gamma,c_x],[0,f_y,c_y],[0,0,1]]).astype(np.float32)

# Getting the rotational and translational matrices (R,t) transforming paper coordinate system --> camera coordinate system using the computed homography and camera intrinsic matrix
def calc_rot_trans_matrices(K,homographies):
    # Initializing the list of matrices (one for each image)
    matrices = []
    for H in homographies:
        # Getting R,t from H,K as specified by Zhang's Camera Calibration Algorithm
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

# Getting the locations of the reprojected corners from paper coordinate system --> camera & pixel coordinate system --> paper coordinate system for a single image
def reproject_corners_single_image(intrinsic_params,base_corners,img_corners,orig_rot_trans_matrix):
    # Getting the list of instrinsic camera parameters from the instrinsic camera matrix (K)
    f_x, f_y, c_x, c_y, gamma, k1, k2 = list(intrinsic_params)
    # Reconstructing the intrinsic camera matrix
    K = np.array([[f_x,gamma,c_x],[0,f_y,c_y],[0,0,1]]).astype(np.float32)
    # 3x3 R,t matrix from paper --> camera coordinate system (this matrix does not consider the effect of Z as the Z coordinate for the chessboard corners are 0 anyway)
    rot_trans_matrix = np.array([orig_rot_trans_matrix[:3,0],orig_rot_trans_matrix[:3,1],orig_rot_trans_matrix[:3,3]]).reshape(3,3).T
    # 3*3 transformation matrix from paper --> pixel coordinate system
    ext_int_matrix = np.dot(K,rot_trans_matrix)
    # List of reprojected corners for the image
    reprojected_corners = []
    # Looping through the corners of the image
    for i in range(img_corners.shape[0]):
        # Converting the ground-truth chessboard corners to different shapes to use each at appropriate places
        base_corner_3D = np.array([base_corners[i,0],base_corners[i,1],0,1]).reshape(4,1)
        base_corner_2D = np.array([base_corners[i,0],base_corners[i,1],1]).reshape(3,1)
        # Transforming the corner from paper --> camera/image coordinate system
        img_plane_proj_corner = np.dot(orig_rot_trans_matrix,base_corner_3D).reshape(4)
        img_plane_proj_corner = img_plane_proj_corner/img_plane_proj_corner[2]
        x, y = img_plane_proj_corner[0], img_plane_proj_corner[1]
        # Calculating the distortion radius
        distortion_radius = ((x)**2 + (y)**2)**0.5
        # Transforming the corner from paper --> pixel coordinate system
        sensor_plane_proj_corner = np.dot(ext_int_matrix,base_corner_2D).reshape(3)
        sensor_plane_proj_corner = sensor_plane_proj_corner/sensor_plane_proj_corner[2]
        u, v = sensor_plane_proj_corner[0], sensor_plane_proj_corner[1]
        # u' and v' represent the reprojected corners onto the image using the pinhole camera model
        u_dash = u + (u-c_x)*(k1*(distortion_radius**2) + k2*(distortion_radius**4))
        v_dash = v + (v-c_y)*(k1*(distortion_radius**2) + k2*(distortion_radius**4))
        reprojected_corners.append(np.array([u_dash,v_dash,0,1]).astype(np.float32).reshape(4,1))
    return reprojected_corners

# Getting the reprojection error for all corners for a single image using the reprojected corners
def reprojection_error_single_image(intrinsic_params,base_corners,img_corners,orig_rot_trans_matrix):
    # Initializing the reprojection error to 0
    reprojection_error = 0
    # Getting the locations of the reprojected corners for the image
    reprojected_corners = reproject_corners_single_image(intrinsic_params,base_corners,img_corners,orig_rot_trans_matrix)
    # Looping through the reprojected corners and summing each of their individual errors to get the total reprojection error for the image
    for i in range(len(reprojected_corners)):
        actual_corner = np.array([img_corners[i][0],img_corners[i][1],0,1]).reshape(4,1)
        error = np.linalg.norm(reprojected_corners[i]-actual_corner)
        reprojection_error += error
    # Scaling the final reprojection error by the number of corners being checked
    return reprojection_error/(SQUARES_X*SQUARES_Y)
    
# Calculating the total reprojection error for all images and summing them up together
def total_reprojection_error(intrinsic_params,base_corners,all_img_corners,rot_trans_matrices):
    # Initializing the list of each individual reprojection error
    reprojection_errors = []
    # Looping through all the images and appending their respective reprojection error to the previosuly initialized list
    for i in range(len(all_img_corners)):
        reprojection_errors.append(reprojection_error_single_image(intrinsic_params,base_corners,all_img_corners[i],rot_trans_matrices[i]))
    return np.array(reprojection_errors)

# Using a least-squares method to optimize the earlier calculated parameters
def optimize_distortion_params(old_intrinsic_params,base_corners,all_img_corners,rot_trans_matrices):
    # Initializing the least-squares optimizer
    ls_params = least_squares(fun=total_reprojection_error,x0=old_intrinsic_params,method="lm",args=[base_corners,all_img_corners,rot_trans_matrices])
    # Listing out the new/optimized camera intrinsic and distortion parameters
    new_intrinsic_params = ls_params.x
    f_x, f_y, c_x, c_y, gamma, k1, k2 = list(new_intrinsic_params)
    # Reconstructing the intrinsic camera and distortion matrices from the optimized parameters
    K = np.array([[f_x,gamma,c_x],[0,f_y,c_y],[0,0,1]]).astype(np.float32)
    Kc = np.array([[k1],[k2]]).astype(np.float32)
    return K, Kc

# Plotting the total reprojection error (for each corner) for each image
def plot_reprojection_errors(intrinsic_params,base_corners,all_img_corners,rot_trans_matrices):
    # Initializing the list of reprojection errors for each image
    img_reprojection_errors = []
    # Looping through all the images and saving their individual reprojection error
    for i in range(len(all_img_corners)):
        img_reprojection_errors.append(reprojection_error_single_image(intrinsic_params,base_corners,all_img_corners[i],rot_trans_matrices[i]))
    # Plotting the reprojection error
    x = range(1,len(img_reprojection_errors)+1)
    y = np.array(img_reprojection_errors)
    plt.title("Reprojection Error for Each Image")
    plt.xlabel("Image Number")
    plt.ylabel("Reprojection Error")
    plt.plot(x,y)
    plt.savefig("results/reprojection_error_graph.png")
    plt.show()

# Overlaying the actual corner positions with the reprojected corner positions for each image
def show_reprojected_corners(imgs,intrinsic_params,base_corners,all_img_corners,rot_trans_matrices):
    # Looping through each image
    for i in range(len(imgs)):
        # Making a deepcopy of the image and obtaining the locations of the reprojected corners
        img = copy.deepcopy(imgs[i])
        reprojected_corners = reproject_corners_single_image(intrinsic_params,base_corners,all_img_corners[i],rot_trans_matrices[i])
        # Marking the locations of the actual corner locations (in red) and the reprojected corners (in blue) over the image
        for j in range(len(reprojected_corners)):
            cv2.circle(img,[int(all_img_corners[i][j][0]),int(all_img_corners[i][j][1])],15,(0,0,255),-1)
            cv2.circle(img,[int(reprojected_corners[j][0,0]),int(reprojected_corners[j][1,0])],9,(255,0,0),-1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.array(img)
        if i==0:
            cv2.imwrite("results/reprojected_corners.png",img)
        plt.imshow(img)
        plt.show()

# Showing the undistorted image  using the intrinsic camera and distortion matrix
def show_undistorted_imgs(imgs,K,Kc):
    # Converting the distortion matrix to a suitable format to undistort the image
    Kc_full = np.array([Kc[0,0],Kc[1,0],0,0]).astype(np.float32)
    # Looping through each of the images
    for i in range(len(imgs)):
        # Undistorting each image and displaying it
        undistorted_img = cv2.undistort(copy.deepcopy(imgs[i]),K,Kc_full)
        undistorted_img = cv2.cvtColor(undistorted_img,cv2.COLOR_BGR2RGB)
        undistorted_img = np.array(undistorted_img)
        if i==0:
            cv2.imwrite("results/undistorted_img.png",undistorted_img)
        plt.imshow(undistorted_img)
        plt.show()

# Wrapper function to get the chessboard corners, obtain and optimize calibration parameters, and plot/display the necessary graphs/results
def calibrate_wrapper(base_img,imgs):
    # Initializing the list of homographies from paper --> camera coordinate system for each image
    homographies = []
    # Obtaining the ground-truth chessboard corner locations
    base_corners = get_corners(copy.deepcopy(base_img))
    # Obtaining the positions of the chessboard corners for each image
    all_img_corners = []
    for img in imgs:
        img_corners = get_corners(copy.deepcopy(img))
        all_img_corners.append(img_corners)
        H = calculate_homography(base_corners,img_corners)
        homographies.append(H)
    # Obtaining the intial estimate of the intrinsic camera matrix and R,t matrices using Zhang's Camera Calibration Algorithm
    V = combined_v_vector(homographies)
    B = B_matrix(V)
    K = intrinsic_matrix(B)
    rot_trans_matrices = calc_rot_trans_matrices(K,homographies)
    old_instrinsic_params = np.array([K[0,0],K[1,1],K[0,2],K[1,2],K[0,1],0,0]).astype(np.float32)
    # Initializing the distortion matrix parameters to 0 (initially)
    Kc = np.array([[0],[0]]).astype(np.float32)
    # Optimizng the camera intrinsic and distortion matrices using the least-squares technique
    K, Kc = optimize_distortion_params(old_instrinsic_params,base_corners,all_img_corners,rot_trans_matrices)
    # Adjusting the R,t matrices using the newly otpimized camera intrinsic and distortion matrices
    rot_trans_matrices = calc_rot_trans_matrices(K,homographies)
    intrinsic_params = np.array([K[0,0],K[1,1],K[0,2],K[1,2],K[0,1],Kc[0,0],Kc[1,0]]).astype(np.float32)
    # Plotting the new reprojection error for each image
    plot_reprojection_errors(intrinsic_params,base_corners,all_img_corners,rot_trans_matrices)
    # Overlaying the reprojected corners (using the pinhole camera model) over the actual corner positions for each image
    show_reprojected_corners(imgs,intrinsic_params,base_corners,all_img_corners,rot_trans_matrices)
    # Displaying the undistorted version of each image by making use of the camera intrinsic and distortion matrices
    show_undistorted_imgs(imgs,K,Kc)
    # Printing the camera instrinsic and distortion matrices to the terminal
    print("Camera Intrinsic Matrix (K):")
    print(K)
    print("*************************************************")
    print("Camera Distortion Matrix (Kc):")
    print(Kc)

def main():
    # Reading the ground-truth/base/refernce chessboard image
    base_img = cv2.imread("checkerboard_pattern.jpg")
    # Reading the multiple views of the chessboard printout
    imgs = load_imgs(CALIBRATION_IMGS_PATH)
    # Performing necessary calibration and obtaining and plotting the required parameters/results
    calibrate_wrapper(base_img,imgs)

if __name__ == "__main__":
    main()