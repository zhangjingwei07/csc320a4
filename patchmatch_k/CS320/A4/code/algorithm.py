# CSC320 Winter 2018
# Assignment 4
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):

    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################

    # find the dimension of the source image
    x, y, k = source_patches.shape[1], source_patches.shape[0], len(f_heap[0][0])

    # try to find the max index that reduce the search radius below 1 pixel.
    max_index = 0
    while (w * (alpha ** max_index) >= 1):
        max_index += 1

    coefficient = np.zeros((max_index + 1, 2)).astype(int)

    for i in range(0, max_index+1):
        coefficient[i, :] = np.array([w * (alpha ** i), w * (alpha ** i)])

    coefficient = np.array([coefficient, ] * k).reshape(((max_index + 1) * k, 2))
    # do the iteration
    # if the iteration number is odd
    if odd_iteration:
        for y_pos in range(0, y):
            for x_pos in range(0, x):
                if propagation_enabled:
                    offset_dict = None
                    if y_pos == 0:
                        if x_pos == 0:
                            break
                        elif x_pos > 0:
                            # get all candidates offset in the list
                            offset_dict = f_coord_dictionary[y_pos][x_pos - 1]
                    else:
                        if x_pos == 0:
                            offset_dict = f_coord_dictionary[y_pos - 1][x_pos]
                        elif x_pos > 0:
                            offset_dict = f_coord_dictionary[y_pos - 1][x_pos].copy()
                            offset_dict.update(f_coord_dictionary[y_pos][x_pos - 1])
                    if offset_dict is not None:
                        # find all entrys in the offset list
                        offset_list = np.array(offset_dict.keys())
                        # get the index in the target source
                        targetImageIndex = offset_list + [y_pos, x_pos]
                        target_idxy = np.clip(targetImageIndex[:, 0], 0, y - 1)
                        target_idxx = np.clip(targetImageIndex[:, 1], 0, x - 1)
                        index_list = np.array([[y_pos, x_pos], ] * offset_list.shape[0])
                        # a list of candidate patches of the source image
                        source_list = np.nan_to_num(source_patches[index_list[:, 0], index_list[:, 1]])
                        # a list of candidate patches of the target image
                        target_list = np.nan_to_num(target_patches[target_idxy, target_idxx])
                        distance_sqaure = np.sum(np.abs(target_list - source_list) * np.abs(target_list - source_list),
                                                 axis=(1, 2))
                        distance_list = np.sqrt(distance_sqaure)

                        # find the max distance in the heap for current position (y_pos x_pos)
                        max_dis = f_heap[y_pos][x_pos][0]
                        while distance_list.min() < -max_dis[0]:
                            min_index = distance_list.argmin()
                            distance = distance_list[min_index]
                            distance_list[min_index] = np.inf
                            offset_i = (target_idxy[min_index] - y_pos, target_idxx[min_index] - x_pos)
                            if offset_i not in f_coord_dictionary[y_pos][x_pos].keys():
                                f_coord_dictionary[y_pos][x_pos].pop(tuple(max_dis[2]), None)
                                f_coord_dictionary[y_pos][x_pos][offset_i] = 0
                                heappushpop(f_heap[y_pos][x_pos], (-distance, _tiebreaker.next(), offset_i))
                                max_dis = f_heap[y_pos][x_pos][0]

                if random_enabled:
                    index_list = np.array([[y_pos, x_pos], ] * ((max_index+1) * k))
                    Ri = np.random.uniform(-1, 1, ((max_index+1) * k, 2))
                    offset_key = np.array(f_coord_dictionary[y_pos][x_pos].keys())
                    offset_list = np.array([offset_key, ] * (max_index+1)).reshape(((max_index + 1) * k), 2)
                    Ui = offset_list + coefficient * Ri
                    targetImageIndex = [y_pos, x_pos] + Ui
                    #print('yyyyyy', y)
                    #print("before cliping ", targetImageIndex[:, 0])
                    target_idxy = np.clip(targetImageIndex[:, 0], 0, y - 1).astype(int)
                    #print("after cliping ", target_idxy)
                    target_idxx = np.clip(targetImageIndex[:, 1], 0, x - 1).astype(int)
                    # a list of candidate patches of the target image
                    target_list = np.nan_to_num(target_patches[target_idxy, target_idxx])
                    # a list of candidate patches of the source image
                    source_list = np.nan_to_num(source_patches[index_list[:, 0], index_list[:, 1]])

                    #print('the size of the patch is ', target_patches.shape)
                    #print('the size of the source is ', source_patches.shape)
                    distance_sqaure = np.sum(np.abs(target_list - source_list) * np.abs(target_list - source_list),
                                         axis=(1,2))
                    distance_list = np.sqrt(distance_sqaure)
                    new_off_y = target_idxy - index_list[:, 0]
                    new_off_x = target_idxx - index_list[:, 1]

                    max_dis = f_heap[y_pos][x_pos][0]
                    while distance_list.min() < -max_dis[0]:
                        min_index = distance_list.argmin()
                        distance = distance_list[min_index]
                        distance_list[min_index] = np.inf
                        offset_i = (new_off_y[min_index], new_off_x[min_index])
                        if offset_i not in f_coord_dictionary[y_pos][x_pos].keys():
                            f_coord_dictionary[y_pos][x_pos].pop(tuple(max_dis[2]), None)
                            f_coord_dictionary[y_pos][x_pos][offset_i] = 0
                            heappushpop(f_heap[y_pos][x_pos], (-distance, _tiebreaker.next(), offset_i))
                            max_dis = f_heap[y_pos][x_pos][0]

    else:
        for y_pos in range(y - 1, -1, -1):
            for x_pos in range(x - 1, -1, -1):
                if propagation_enabled:
                    offset_dict = None
                    if y_pos == y - 1:
                        if x_pos == x - 1:
                            pass
                        elif x_pos < x - 1:
                            offset_dict = f_coord_dictionary[y_pos][x_pos + 1]
                    elif y_pos < y -1:
                        if x_pos == x - 1:
                            offset_dict = f_coord_dictionary[y_pos + 1][x_pos]
                        elif x_pos < x - 1:
                            offset_dict = f_coord_dictionary[y_pos + 1][x_pos].copy()
                            offset_dict.update(f_coord_dictionary[y_pos][x_pos + 1])
                    if offset_dict is not None:
                        # find all entrys in the offset list
                        offset_list = np.array(offset_dict.keys())
                        # get the index in the target source
                        targetImageIndex = offset_list + [y_pos, x_pos]
                        target_idxy = np.clip(targetImageIndex[:, 0], 0, y - 1)
                        target_idxx = np.clip(targetImageIndex[:, 1], 0, x - 1)
                        index_list = np.array([[y_pos, x_pos], ] * offset_list.shape[0])
                        # a list of candidate patches of the source image
                        source_list = np.nan_to_num(source_patches[index_list[:, 0], index_list[:, 1]])
                        # a list of candidate patches of the target image
                        target_list = np.nan_to_num(target_patches[target_idxy, target_idxx])
                        distance_sqaure = np.sum(np.abs(target_list - source_list) * np.abs(target_list - source_list),
                                                 axis=(1, 2))
                        distance_list = np.sqrt(distance_sqaure)

                        # find the max distance in the heap for current position (y_pos x_pos)
                        max_dis = f_heap[y_pos][x_pos][0]
                        while distance_list.min() < -max_dis[0]:
                            min_index = distance_list.argmin()
                            distance = distance_list[min_index]
                            distance_list[min_index] = np.inf
                            offset_i = (target_idxy[min_index] - y_pos, target_idxx[min_index] - x_pos)
                            if offset_i not in f_coord_dictionary[y_pos][x_pos].keys():
                                f_coord_dictionary[y_pos][x_pos].pop(tuple(max_dis[2]), None)
                                f_coord_dictionary[y_pos][x_pos][offset_i] = 0
                                heappushpop(f_heap[y_pos][x_pos], (-distance, _tiebreaker.next(), offset_i))
                                max_dis = f_heap[y_pos][x_pos][0]

                if random_enabled:
                    index_list = np.array([[y_pos, x_pos], ] * ((max_index + 1) * k))
                    Ri = np.random.uniform(-1, 1, ((max_index + 1) * k, 2))
                    offset_key = np.array(f_coord_dictionary[y_pos][x_pos].keys())
                    offset_list = np.array([offset_key, ] * (max_index + 1)).reshape(((max_index + 1) * k), 2)
                    Ui = offset_list + coefficient * Ri
                    targetImageIndex = [y_pos, x_pos] + Ui
                    # print('yyyyyy', y)
                    # print("before cliping ", targetImageIndex[:, 0])
                    target_idxy = np.clip(targetImageIndex[:, 0], 0, y - 1).astype(int)
                    # print("after cliping ", target_idxy)
                    target_idxx = np.clip(targetImageIndex[:, 1], 0, x - 1).astype(int)
                    # a list of candidate patches of the target image
                    target_list = np.nan_to_num(target_patches[target_idxy, target_idxx])
                    # a list of candidate patches of the source image
                    source_list = np.nan_to_num(source_patches[index_list[:, 0], index_list[:, 1]])

                    # print('the size of the patch is ', target_patches.shape)
                    # print('the size of the source is ', source_patches.shape)
                    distance_sqaure = np.sum(np.abs(target_list - source_list) * np.abs(target_list - source_list),
                                             axis=(1, 2))
                    distance_list = np.sqrt(distance_sqaure)
                    new_off_y = target_idxy - index_list[:, 0]
                    new_off_x = target_idxx - index_list[:, 1]

                    max_dis = f_heap[y_pos][x_pos][0]
                    while distance_list.min() < -max_dis[0]:
                        min_index = distance_list.argmin()
                        distance = distance_list[min_index]
                        distance_list[min_index] = np.inf
                        offset_i = (new_off_y[min_index], new_off_x[min_index])
                        if offset_i not in f_coord_dictionary[y_pos][x_pos].keys():
                            f_coord_dictionary[y_pos][x_pos].pop(tuple(max_dis[2]), None)
                            f_coord_dictionary[y_pos][x_pos][offset_i] = 0
                            heappushpop(f_heap[y_pos][x_pos], (-distance, _tiebreaker.next(), offset_i))
                            max_dis = f_heap[y_pos][x_pos][0]

    #############################################

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    f_heap = None
    f_coord_dictionary = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # initial head and coord_dictionary to empty list
    f_heap = []
    f_coord_dictionary = []
    # get y size
    y = source_patches.shape[0]
    # get x size
    x = source_patches.shape[1]
    coordinate = make_coordinates_matrix((y, x))
    # get k size
    k = f_k.shape[0]
    # try to get the position in the target image
    target_position = f_k + np.array([coordinate, ] * k)
    target_y = np.clip(target_position[:, :, :, 0], 0, y -1)
    target_x = np.clip(target_position[:, :, :, 1], 0, x - 1)
    # get the source patch
    k_coor = np.array([coordinate, ] * k)
    source_list = np.nan_to_num(source_patches[k_coor[:, :, :, 0], k_coor[:, :, :, 1]])
    # get the target patch
    target_list = np.nan_to_num(target_patches[target_y, target_x])
    # compute the distance between source_list and target_list
    distance_list = np.sum(np.linalg.norm(target_list - source_list, axis=-1), axis=-1)
    for y_pos in range(y):
        Rowheap = []
        Rowcoord = []
        for x_pos in range(x):
            heap = []
            dict = {}
            for i in range(k):
                heappush(heap, (-distance_list[i, y_pos, x_pos], _tiebreaker.next(), f_k[i, y_pos, x_pos]))
                dict[tuple(f_k[i, y_pos, x_pos])] = 0
            # put heap into the row heap
            # put dic into the row dictionary
            Rowheap.append(heap)
            Rowcoord.append(dict)
        f_heap.append(Rowheap)
        f_coord_dictionary.append(Rowcoord)
    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    # get a copy of the f_heap
    # get k, x, y of the matrix
    k = len(f_heap[0][0])
    x = len(f_heap[0])
    y = len(f_heap)
    # initialize f_k and D_k
    f_k = np.zeros((k, y, x, 2))
    D_k = np.zeros((k, y, x))
    for y_pos in range(0, y, 1):
        for x_pos in range(0, x, 1):
            for i in range(0, k, 1):
                # get next best element
                element = f_heap[y_pos][x_pos][i]
                # get the distance
                distance = -element[0]
                # get the next best nnf in the heap
                next_best_nnf = element[2]
                f_k[i, y_pos, x_pos] = next_best_nnf
                D_k[i, y_pos, x_pos] = distance

    #############################################

    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    #denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    # get dimension of the image
    y, x = target.shape[0], target.shape[1]
    k = len(f_heap[0][0])
    denoised = np.zeros((y, x, 3))
    for y_pos in range(0, y, 1):
        for x_pos in range(0, x, 1):
            # first get negative difference in the heap into a list
            sim_list = np.array([item[0] for item in f_heap[y_pos][x_pos]])
            # compute the exponent in the equation
            exponent = sim_list.astype(float) / (h ** 2)
            # apply the exponent to the natural number e
            unnormalized_weight = np.exp(exponent)
            normalized_weight = unnormalized_weight / np.sum(unnormalized_weight)


            # get the offset
            offset_list = np.array([item[2] for item in f_heap[y_pos][x_pos]])
            # get the index in the target image
            target_idx = offset_list + [y_pos, x_pos]
            # get the y and x index
            target_idxy = np.clip(target_idx[:, 0], 0, y - 1)
            target_idxx = np.clip(target_idx[:, 1], 0, x - 1)
            target_pixel = target[target_idxy, target_idxx]

            # extend to 3 channels
            three_channel = np.dstack((normalized_weight, normalized_weight, normalized_weight))[0]
            result_sum = np.sum(three_channel * target_pixel, axis=0)
            # assign the value to the denoised image
            denoised[y_pos, x_pos] = result_sum
    # change the array type to unit8
    denoised = denoised.astype(np.uint8)
    #############################################

    return denoised




#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################


#############################################



# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################
    target_index = make_coordinates_matrix(target.shape) + f
    x = np.clip(target_index[:, :, 0], 0, target.shape[0] - 1).astype(int)
    y = np.clip(target_index[:, :, 1], 0, target.shape[1] - 1).astype(int)
    rec_source = target[x, y]
    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
