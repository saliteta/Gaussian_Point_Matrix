import numpy as np
from plyfile import PlyData

class Points:
    def __init__(self, ply_file) -> None:
        self.ply_file = ply_file
        self.load()

    def __getitem__(self, index):
        return self.location[index], self.color[index], 

    def __len__(self):
        return len(self.location)
    
    def load(self):
        vertex_data = PlyData.read(self.ply_file)['vertex']
        self.location = np.array([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
        self.color = np.array([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T

        print(f'loading of file {self.ply_file} accomplished with {len(self)} points have been loaded')

class Gaussian:
    def __init__(self, Gaussian_file) -> None:
        self.gaussian_file = Gaussian_file
        self.load_gaussian()

    def __getitem__(self, index):
        return self.means[index], self.scales[index], self.rotation[index], self.color[index]

    def __len__(self):
        return len(self.means)


    def sh2rgb(self, sh:np.ndarray):
        '''
        This will turn sh1 to rgb color
        According to this repo: 
        https://github.com/francescofugazzi/3dgsconverter/blob/main/gsconverter/utils/utility.py#L68
        And this issue: 
        https://github.com/graphdeco-inria/gaussian-splatting/issues/485
        We find that 
        value=(value+1)*128 (I have no idea why this is correct)
        '''
        colors = (sh + 1) * 127.5
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        return colors

    def load_gaussian(self):
        '''
        This will load the Gaussian Splatting
        it will contains:
        - means (x, y, z)
        - scales (x, y, z)
        - rotation (q1, q2, q3, q4)
        - color (r, g, b)
        '''
        vertex_data = PlyData.read(self.gaussian_file)['vertex']
        self.means = np.array([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T 
        self.scales = np.array([vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2']]).T
        self.rotation = np.array([vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3']]).T
        self.color = self.sh2rgb(np.array([vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2']]).T)

        print(f'loading of file {self.gaussian_file} accomplished with {len(self)} Gaussians have been loaded')

if __name__ == "__main__":
    Gaussian('gaussian_splatting.ply')