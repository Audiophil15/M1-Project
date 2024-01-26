from .image_base import Image
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from .utils import bilinear_interpolate, pad_array, coeffs
import numpy as np
import copy
import cv2

class ImageCorruption(Image):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.mb_kernel_size = 15
        self.rb_kernel_size = 5

    ############################################# Corruption Functions ######################################################   
    def symmetry_transform(self, axis):
        """
        Implementation of symmetry transform corruptions.

        :param axis: line along which a symmetric mapping should be drawn ['x', 'y', 'diagonal']
        :type axis: str

        :return: corrupted image
        :rtype: np.array
        """
        corrupted_image = np.full_like(self.image, fill_value=255)
        if axis == 'y':
            corrupted_image = np.flip(self.image, axis=1)
        elif axis == 'x':
            corrupted_image = np.flip(self.image, axis=0)
        elif axis == 'diagonal':
            corrupted_image = np.transpose(self.image)
        else:
            raise ValueError('You can choose only x, y and diagonal symmetry transform')
        
        return corrupted_image
        
    def create_squares(self, color, coordinates):
        '''
        coordinates in format start_height, start_width, height, width
        '''
        """
        Implementation of corruption with colorful squares in certain coordinates .

        :param axis: line along which a symmetric mapping should be drawn ['x', 'y', 'diagonal']
        :type axis: str

        :return: corrupted image
        :rtype: np.array
        """
        corrupted_image = copy.deepcopy(self.image)
        if color == 'white':
            if self.normalized:
                corrupted_image[coordinates[0] : coordinates[0] + coordinates[2], 
                           coordinates[1] : coordinates[1] + coordinates[3]] = 1.
            else:
                corrupted_image[coordinates[0] : coordinates[0] + coordinates[2], 
                           coordinates[1] : coordinates[1] + coordinates[3]] = 255
            
        elif color == 'black':
            if self.normalized:
                corrupted_image[coordinates[0] : coordinates[0] + coordinates[2], 
                           coordinates[1] : coordinates[1] + coordinates[3]] = 0.
            else:
                corrupted_image[coordinates[0] : coordinates[0] + coordinates[2], 
                           coordinates[1] : coordinates[1] + coordinates[3]] = 0.
        
        else:
            raise ValueError('You can choose only "black" or "white" color')
        
        return corrupted_image

    def rotate_image_with_interpolation(self, angle):
        """
        """
        # Calculate the center
        center_x, center_y = self.width / 2, self.height / 2
        # Create the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
        # Create an empty image for the output with white background
        corrupted_image = np.full_like(self.image, fill_value=255)
        # Perform the rotation
        for y in range(self.height):
            for x in range(self.width):
                # Apply the rotation matrix
                new_x, new_y = np.dot(rotation_matrix, np.array([x, y, 1]))
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    corrupted_image[y, x] = bilinear_interpolate(self.image, new_x, new_y)

        return corrupted_image
    
    def motion_blur(self):
        """
        """
        kernel = np.zeros((self.mb_kernel_size, self.mb_kernel_size))
        intensity = 1.0 / self.mb_kernel_size

        for i in range(self.mb_kernel_size):
            kernel[i, :i+1] = intensity
        kernel /= kernel.sum()

        corrupted_image = self.fft_convolution(kernel)

        return corrupted_image
    
    def radial_blur(self, R=5):
        """
        """
        center_x, center_y = self.rb_kernel_size // 2, self.rb_kernel_size // 2 
        rb_kernel = self.radial_blur_kernel(center_x, center_y, R)
        
        corrupted_image = self.fft_convolution(rb_kernel)

        return corrupted_image
    
    def elastic_twist_local(self, max_angle, center_x, center_y, radius, distortion_radius):
        """
        """
        deformation_map_x = np.zeros_like(self.image, dtype=np.float32)
        deformation_map_y = np.zeros_like(self.image, dtype=np.float32)

        corrupted_image = copy.deepcopy(self.image)
        for y in range(self.height):
            for x in range(self.width):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                max_distance = radius

                if distance < radius:
                    angle = (1 - np.power(distance / max_distance, 2)) * max_angle
                    angle_rad = np.deg2rad(angle)
                    stretch = 1.0

                    if distance > (radius - distortion_radius):
                        blur_radius = (distance - (radius - distortion_radius)) / distortion_radius
                        corrupted_image[y, x] = gaussian_filter(corrupted_image[y, x], sigma=blur_radius)

                    deformation_map_x[y, x] = center_x + (x - center_x) * np.cos(angle_rad) * stretch - (y - center_y) * np.sin(angle_rad) * stretch
                    deformation_map_y[y, x] = center_y + (x - center_x) * np.sin(angle_rad) * stretch + (y - center_y) * np.cos(angle_rad) * stretch
                else:
                    deformation_map_x[y, x] = x
                    deformation_map_y[y, x] = y

        corrupted_image = cv2.remap(corrupted_image, 
                                    deformation_map_x, 
                                    deformation_map_y, 
                                    cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=255)

        return corrupted_image
    
    def weak_pressure(self, x0, y0, a = 0.1, b = 0.5, fading_range = 0.1, angle = 0., inplace = False):
        """
        """
        corrupted_image = copy.deepcopy(self.image)
        c = coeffs(self.image, np.array([x0, y0]), [0, self.height], [0, self.width], a, b, fading_range, angle)
        for i in range(self.height) :
            for j in range(self.width) :
                l = c[i, j]
                if self.normalized :
                    corrupted_image[i,j] = min(self.image[i, j]+(1-l), 1)
                else :
                    corrupted_image[i,j] = min(self.image[i, j]+int((1-l)*255), 255)
        if inplace :
            self.image = corrupted_image
        return corrupted_image
    
    ########################################## Supporting functions necessary for the realisation of the main corruptions ###################################################
    def cast_to_float(self):
        """
        """
        if not self.normalized:
            self.image = self.image.astype(np.float64) / 255.
            self.normalized = True
        else:
            print('Already normalized')

    def bilinear_interpolate(self, x, y):
        """
        """
        # Calculate the coordinates of the four surrounding pixels
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, self.width - 1), min(y1 + 1, self.height - 1)
        # Calculate the differences
        dx, dy = x - x1, y - y1
        # Interpolate
        interpolated = np.zeros(self.channels)
        for c in range(self.channels):
            interpolated[c] = (self.image[y1, x1, c] * (1 - dx) * (1 - dy) +
                            self.image[y1, x2, c] * dx * (1 - dy) +
                            self.image[y2, x1, c] * (1 - dx) * dy +
                            self.image[y2, x2, c] * dx * dy)
        return interpolated
    
    def naive_convolution(self, kernel):
        """
        A naive implementation of convolution with a symmetric kernel.

        :param image: input matrix on which the convolution will be performed
        :type image: np.array
        :param kernel: symmetric kernel
        :type kernel: np.array

        :return: convolution result
        :rtype: np.array
        """
        hk, wk = kernel.shape

        pad_size = (hk - 1) // 2
        image_padded = pad_array(self.image, pad_size)

        out = np.zeros(shape=self.image.shape)
        for row in range(self.height):
            for col in range(self.width):
                for i in range(hk):
                    for j in range(wk):
                        out[row, col] += image_padded[row + i, col + j] * kernel[i, j]
        return out
    
    def fft_convolution(self, kernel):
        """
        Convergence algorithm using Fourier transform as optimisation.

        :param image: input matrix on which the convolution will be performed
        :type image: np.array
        :param kernel: kernel matrix
        :type kernel: np.array

        :return: convolution result
        :rtype: np.array
        """
        k, l = kernel.shape

        p = self.height + k - 1
        q = self.width + l - 1

        padded_image = np.pad(self.image, ((0, p - self.height), (0, q - self.width)), mode='constant')
        padded_kernel = np.pad(kernel, ((0, p - k), (0, q - l)), mode='constant')

        fft_image = fft2(padded_image)
        fft_kernel = fft2(padded_kernel)
        fft_result = fft_image * fft_kernel

        result = np.real(ifft2(fft_result))[:self.height, :self.width]

        return result 
    
    def radial_blur_kernel(self, xc, yc, R):
        """
        """
        kernel = np.zeros((self.rb_kernel_size, self.rb_kernel_size))
        for i in range(self.rb_kernel_size):
            for j in range(self.rb_kernel_size):
                distance_squared = (i - xc) ** 2 + (j - yc) ** 2
                energy_decay = 1 - distance_squared / R ** 2
                kernel[i, j] = np.maximum(0, energy_decay)

        normalized_kernel = kernel / np.sum(kernel)
        return normalized_kernel