from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KDTree
import numpy as np
from utils import Points, Gaussian


class Coverage_matrix:
    def __init__(self, gaussian_name: str, point_cloud_name: str, logging_file_name:str,gaussian_spaltting: Gaussian, lidar_point_cloud: Points, color_threshold: float, k = 5) -> None:
        self.gaussian_spaltting = gaussian_spaltting
        self.gaussian_occupancy = np.zeros(len(self.gaussian_spaltting), dtype=np.int8)
        self.lidar_point_cloud = lidar_point_cloud
        self.points_occupancy = None
        self.color_threshold = color_threshold
        self.tree = KDTree(self.gaussian_spaltting.means)
        self.k = k

        ## logging
        self.gaussian_name = gaussian_name
        self.point_cloud_name = point_cloud_name
        self.logging_file_name = logging_file_name

    def nearest_index(self):
        '''
            Given the nearest up to k different elipsoids index and the distance to them
            index is the index of ellipoids
        '''
        _, indices = self.tree(self.lidar_point_cloud, k = self.k)
        return indices 

    def init_result_array(n_points):
        # Initialize a structured array to hold an integer index and three int8 values for the color
        dtype = [('index', int), ('color_diff', 'float16')]
        result_array = np.full(n_points, (-1, -1), dtype=dtype)  # Default: index -1, color black
        return result_array

    def check_coverage(self, point_location, mean, quat, scale, color):
        # only for a single judgement
        '''
            To check given points is in or not in one elipoids
            We need to convert the coordinates to relative coordinates (get vector)
            and use rotation and scales to determine the vector is inside or not
        '''

        '''
            Notice that rotate the ellipsoids is the same as inversly rotate the point
        '''
        p = point_location - mean
        rotation = R.from_quat(quat)
        p_rotated = rotation.inv().apply(p)
        scale_inv = 1 / np.array(scale)
        p_scaled = p_rotated * scale_inv
        if np.dot(p_scaled, p_scaled) <= 1:
            return True
        return False
    
    def calculate_color_difference(self, point_color, gaussian_color):
        '''
            We are using the L2 norm to calculate the difference between two color
            We might change to use L1 or L-Infi
        '''
        return np.linalg.norm(point_color, gaussian_color)

    def process_points(self, data):
        # Unpack all data necessary for processing a single point against all ellipsoids
        point_index, ellipsoids_index = data
        for index in ellipsoids_index:
            mean, quat, scale, gaussian_color = self.gaussian_spaltting[index]
            point_location, point_color = self.lidar_point_cloud[index]
            covered = self.check_coverage(point_location, mean, quat, scale)
            if covered:
                return point_index, index, self.calculate_color_difference(point_color, gaussian_color)
        return point_index, -1, -1
        # return point index
        # return matched ellipsoid index no match then return -1
        # return rgb color for coherent judgement

    def parallel_coverage(self, gaussian_index):
        # Prepare a list of tasks with all necessary data for each point-ellipsoid check

        n_points = len(self.lidar_point_cloud)
        result_array = self.init_result_array(n_points)
        tasks = [(i, gaussian_index[i]) for i in range(n_points)]


        with ProcessPoolExecutor() as executor:
            for point_index, ellipsoid_index, color in executor.map(self.process_points, tasks):
                result_array[point_index] = (ellipsoid_index, color)
        
        return result_array

        # Process results
        # This is where you'd handle the results, e.g., determining which points are covered by which ellipsoid

    def preservance_score(self):
        '''
            When we have the coverage result we can use how many -1 is contained
            the number of -1 over the number of points
            rememeber to set the occupancy to points
        '''
        not_mathcing_points_number = np.count_nonzero(self.points_occupancy == -1)
        preservance_score = (1 - not_mathcing_points_number/len(self.lidar_point_cloud))*100 
        return preservance_score

    def clearance_score(self):
        '''
            set Gaussian Splatting According to the points
            count how many zero we have
        '''
        gaussian_matching_indices = self.points_occupancy[self.points_occupancy>=0]
        self.gaussian_occupancy[gaussian_matching_indices] = 1
        matching_number = np.count_nonzero(self.gaussian_occupancy)
        clearance_score = matching_number/len(self.gaussian_occupancy) * 100
        return clearance_score 

    def coherent_score(self, threshold = 10.0):
        '''
            compare the color one by one
            2 to 24 for L2 norm could be select as a good threshould
            In the data we have, it could either be -1 or a positive number
        '''

        matching_points_color_difference = self.color_difference[self.points_occupancy>=0]
        error_points_number = len(matching_points_color_difference[matching_points_color_difference>threshold])
        coherent_score = 1 - (error_points_number/len(matching_points_color_difference)) * 100
    
        return coherent_score

    def coverage_checking(self):
        '''
            There are overall three matrices to measure if point cloud is align with the
            Gaussian Splatting
            - Point Cloud (AND) Gaussian Splatting / Point Cloud: Preservance Score 
            The higher the score, larger proportion will be cover in the Gaussian Splatting
            - Point Cloud (And) Gaussian Splatting / Gaussian Splatting: Clearance Score
            The higher the score is, the lower unneccesary Gaussian Splatting we have
            - Same Color point(Within a threshold) / Point Cloud (AND) Gaussian Splatting: Coherence Score
            The higher the score is, the better the overlapping region is closer to each other
        '''

        nearest_index_gaussian = self.nearest_index() # (points, k)
        coverage_result = self.parallel_coverage(nearest_index_gaussian) 
        # 2 elements, first element is the matching ellipsoids indices
        # second element is the matching elliposids color difference in L2
        self.points_occupancy = coverage_result[0]
        self.color_difference = coverage_result[1]
        preservance = self.preservance_score()
        clearance = self.clearance_score()
        coherence = self.coherent_score()

        self.logging_result(preservance, clearance, coherence)

    def logging_result(self, preservance, clearance, coherence):
        '''
            We need to log everything
            filename is previously specified
        '''
        print(f"checking difference between point cloud: {self.point_cloud_name} and Gaussian Splatting {self.gaussian_name}")
        print(f"preservance_score: {preservance}%")
        print(f"clearance_score: {clearance}%")
        print(f"coherent_score: {coherence}%")
        print(f"logging result can be checked in {self.logging_file_name}")

        with open(self.logging_file_name) as f: 
            f.writelines(f"checking difference between point cloud: {self.point_cloud_name} and Gaussian Splatting {self.gaussian_name}")
            f.writelines(f"preservance_score: {preservance}%")
            f.writelines(f"clearance_score: {clearance}%")
            f.writelines(f"coherent_score: {coherence}%")
