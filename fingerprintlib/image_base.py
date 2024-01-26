import cv2
import numpy as np

class Image:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.height, self.width = self.image.shape
        self.channels = 1 # default param for grayscale
        self.normalized = False

    def save_image(self, output_path):
        """
        Saving image in certain directory.

        :param output_path: path on which user wants to save image
        :type output_path: str

        :return: None
        :rtype: None
        """
        if self.normalized:
            self.image = (self.image * 255).astype(np.uint8)
            self.normalized = False
        cv2.imwrite(output_path, cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR))
        
    def min_max_intensity(self):
        """
        Calculate minimum and maximum intensity of the image that owns the class.

        :return: tuple with statistics results
        :rtype: int, int
        """
        min_val = np.min(self.image)
        max_val = np.max(self.image)
        return min_val, max_val
    
    def matrix_to_vector(self):
        """
        Transform image representation from matrix to vector.

        :return: image presented with vector
        :rtype: np.array
        """
        m = np.flipud(self.image)
        output = np.zeros(self.height * self.width, dtype=m.dtype)
        for i, row in enumerate(m):
            st = i * self.width
            nd = st + self.width
            output[st:nd] = row

        return output
    
    def vector_to_matrix(self, vect, matrix_shape):
        """
        Transform image representation from vector to matrix.

        :param vect: vector to be updated to the matrix
        :type vect: np.array
        :param matrix_shape: shape of future matrix (shape[0] * shape[1] = len(vect))
        :type matrix_shape: (int, int)

        :return: image presented with matrix
        :rtype: np.array
        """
        o_h, o_w = matrix_shape
        output = np.zeros(matrix_shape, dtype=vect.dtype)
        for i in range(o_h):
            st = i * o_w
            nd = st + o_w
            output[i, :] = vect[st: nd]

        output = np.flipud(output)
        return output

        

    




