from .image_base import Image
import copy
import numpy as np
from scipy.linalg import toeplitz
from .utils import create_mask, create_patches

class ImageRestauration(Image):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.restoration_mask = None
        self.patch_size = None
        self.patches = []

    def convolution_matrix(self, kernel):
        """
        Calculate sparse convolution matrix from kernel to perform conv operation by matrix multiplication.

        :param kernel: kernel-filter for convolution
        :type kernel: np.array

        :return: convolution matrix which is represented by the Toeplitz doubly block matrices.
        :rtype: np.array
        """
        k_row_num, k_col_num = kernel.shape

        out_row_num = self.height + k_row_num - 1
        out_col_num = self.width + k_col_num - 1

        zero_padded_k = np.pad(kernel, ((out_row_num - k_row_num, 0), (0, out_col_num - k_col_num)), 'constant', constant_values=0)

        toeplitz_list = []
        for i in range(zero_padded_k.shape[0] - 1, -1, -1):
            c = zero_padded_k[i, :]
            r = np.r_[c[0], np.zeros(self.width - 1)]

            toeplitz_m = toeplitz(c,r)
            toeplitz_list.append(toeplitz_m)

        c = range(1, zero_padded_k.shape[0] + 1)
        r = np.r_[c[0], np.zeros(self.height - 1, dtype=int)]

        doubly_indices = toeplitz(c, r)
        
        h = toeplitz_list[0].shape[0] * doubly_indices.shape[0]
        w = toeplitz_list[0].shape[1] * doubly_indices.shape[1]
        doubly_blocked = np.zeros([h, w])

        b_h, b_w = toeplitz_list[0].shape
        for i in range(doubly_indices.shape[0]):
            for j in range(doubly_indices.shape[1]):
                start_i, start_j = i * b_h, j * b_w
                end_i, end_j = start_i + b_h, start_j + b_w
                doubly_blocked[start_i : end_i, start_j : end_j] = toeplitz_list[doubly_indices[i, j] - 1]
        
        return doubly_blocked
    
    def blur_restauration(self, kernel, solver, lam=0.3):
        """
        Perform convolution with sparse Toeplitz matrcies and image in vector representation.

        :param kernel: kernel-filter for convolution
        :type kernel: np.array
        :param solver: type of linear equations solver ['ill-posed', 'well-posed']
        :type solver: str
        :param lam: lamdba hyperparametr for well-posed task
        :type lam: int
        
        :return: convolution result
        :rtype: np.array
        """
        image_vector = self.matrix_to_vector()
        matrix_from_kernel = self.convolution_matrix(kernel)

        if solver == 'ill-posed':
            clean_predict = np.linalg.lstsq(matrix_from_kernel.T, image_vector, rcond=None)
        elif solver == 'well-posed':
            zero_vector = np.zeros((3600,))
            zero_eye = np.eye(3600)
            well_image_vector = np.concatenate((image_vector, zero_vector))
            well_matrix = np.block([[matrix_from_kernel.T],[lam * zero_eye]])
            clean_predict = np.linalg.lstsq(well_matrix, well_image_vector, rcond=None)

        restored_image = self.vector_to_matrix(clean_predict[0], (60, 60))

        return restored_image

    def init_pressure_restore(self, *args):
        """
        Initialization of helper values to perform restoration on pressured images.
        
        :return: None
        :rtype: None
        """
        self.patches, self.patch_size = create_patches(self.image, size=9)
        self.restoration_mask = create_mask(self.image, *args)
    
    def presssure_restore(self):
        """
        Realisation of image restoring after pressure simulation.
        
        :return: corrupted image
        :rtype: np.array
        """
        self.init_pressure_restore([91,106], [115,130])
        if (self.restoration_mask.size) :
            output = copy.deepcopy(self.image)
            mask_dict = {}
            for i in range(self.height) :
                for j in range(self.width) :
                    if self.restoration_mask[i, j] :
                        mask_dict[(i, j)] = np.sum(self.restoration_mask[max(0, i-self.patch_size//2) : min(self.height, i+self.patch_size//2+1), 
                                                       max(0, j-self.patch_size//2) : min(self.width, j+self.patch_size//2+1)])
            mask_dict = dict(sorted(mask_dict.items(), key=lambda it:it[1]))

            while len(mask_dict) :
                x, y = list(mask_dict.keys())[0]
                xmin = max(0,x-self.patch_size//2)
                xmax = min(self.height,x+self.patch_size//2+1)
                ymin = max(0,y-self.patch_size//2)
                ymax = min(self.width,y+self.patch_size//2+1)
                imgtmp = self.image[xmin: xmax, ymin: ymax]
                msktmp = self.restoration_mask[xmin: xmax, ymin: ymax]
                d = 255**2
                best_sample = np.empty(0)
                for sample in self.patches :
                    n = np.sum(np.abs(sample-imgtmp)[np.logical_not(msktmp)])
                    if n < d :
                        d = n
                        best_sample = sample
                if best_sample.size != 0 :
                    output[x, y] = best_sample[self.patch_size//2, self.patch_size//2]
                    self.restoration_mask[x, y] = True
                    mask_dict.pop((x, y))
                    for key in mask_dict :
                        i, j = key
                        mask_dict[key] = np.sum(self.restoration_mask[max(0, i-self.patch_size//2) : min(self.height, i+self.patch_size//2+1), 
                                                    max(0, j-self.patch_size//2) : min(self.width, j+self.patch_size//2+1)])
                    mask_dict = dict(sorted(mask_dict.items(), key=lambda it:it[1]))
            return output
        return self.image