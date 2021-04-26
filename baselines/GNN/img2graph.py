import numpy as np
from scipy import sparse

###############################################################################
# From an image to a graph

def _make_edges(n_x, n_y):
    """
        Returns a list of edges and edge weights for a 2D image.

        Parameters
        ----------
        n_x : int
            The size of the grid in the x direction.
        n_y : int
            The size of the grid in the y direction.
    """
    vertices = np.arange(n_x * n_y).reshape((n_x, n_y))
    # vertices [[0, 1, 2, ..., 27],
    #           [28,29,30,..., 55],
    #           [                ]]
    ########### edges ###########
    edges_right = np.vstack((vertices[:, :-1].ravel(),
                             vertices[:, 1:].ravel()))  # edges: ((0,1), (1,2), ...)
    edges_left = np.vstack((vertices[:, 1:].ravel(),
                            vertices[:, :-1].ravel()))  # edges: ((1,0), (2,1), ...)
    edges_down = np.vstack((vertices[:-1].ravel(),
                            vertices[1:].ravel()))      # edges: ((0,28), (1,29), ...)
    edges_up = np.vstack((vertices[1:].ravel(),
                          vertices[:-1].ravel()))  # edges: ((28,0), (29,1), ...)
    edges_diag_down_right = np.vstack((vertices[:-1, :-1].ravel(),
                                       vertices[1:, 1:].ravel())) # edges: ((0,29), (1,30), ...)
    edges_diag_up_left = np.vstack((vertices[1:, 1:].ravel(),
                                      vertices[:-1, :-1].ravel()))  # edges: ((29,0), (30,1), ...)
    edges_diag_up_right = np.vstack((vertices[1:, :-1].ravel(),
                                     vertices[:-1, 1:].ravel()))  # edges: ((28,1), (29,2), ...)
    edges_diag_down_left = np.vstack((vertices[1:, :-1].ravel(),
                                     vertices[:-1, 1:].ravel()))  # edges: ((1,28), (2,29), ...)
    #################################

    ########### weights ###########
    # edge weights ((x_j-x_i, y_j-y_i))
    weights_right = np.array([[0, 1]] * edges_right.shape[1])   # ((0,1),(0,1),...)
    weights_left  = np.array([[0, -1]] * edges_left.shape[1])   # ((0,-1),(0,-1),...)
    weights_down  = np.array([[1, 0]] * edges_down.shape[1])    # ((1,0),(1,0),...)
    weights_up    = np.array([[-1, 0]] * edges_up.shape[1])     # ((-1,0),(-1,0),...)
    weights_diag_down_right = np.array([[1, 1]] * edges_diag_down_right.shape[1])   # ((1,1),  (1,1),...)
    weights_diag_up_left    = np.array([[-1, -1]] * edges_diag_up_left.shape[1])    # ((-1,-1),(-1,-1),...)
    weights_diag_up_right   = np.array([[-1, 1]] * edges_diag_up_right.shape[1])    # ((-1,1), (-1,1),...)
    weights_diag_down_left  = np.array([[1, -1]] * edges_diag_down_left.shape[1])   # ((1,-1), (1,-1),...)
    #################################
    # shape [2, num_edges]
    edges = np.hstack((edges_right, edges_left, edges_down, edges_up,
                       edges_diag_down_right, edges_diag_up_left,
                       edges_diag_up_right, edges_diag_down_left))
    # shape [num_edges, num_edge_feats(2)]
    edge_weights = np.vstack((weights_right, weights_left, weights_down, weights_up,
                              weights_diag_down_right, weights_diag_up_left,
                              weights_diag_up_right, weights_diag_down_left))
    return edges, edge_weights

# not used (keeping here just in case it is required)
def _compute_gradient_3d(edges, img):
    _, n_y, n_z = img.shape
    gradient = np.abs(img[edges[0] // (n_y * n_z),
                      (edges[0] % (n_y * n_z)) // n_z,
                      (edges[0] % (n_y * n_z)) % n_z] -
                      img[edges[1] // (n_y * n_z),
                      (edges[1] % (n_y * n_z)) // n_z,
                      (edges[1] % (n_y * n_z)) % n_z])
    return gradient

def _mask_edges_weights(mask, edges, edge_weights=None):
    """
        Apply a mask to edges (weighted or not)
        Parameters
        ----------
        mask: np.ndarray
            binary matrix with zeros at coords to remove
            shape (img_height, img_width)
        edges: np.ndarray
            edges matrix of shape [2, num_edges]
        weights: np.ndarray
            edges matrix of shape [num_edges, num_edge_feats]
    """
    inds = np.arange(mask.size)
    inds = inds[mask.ravel()]
    ind_mask = np.logical_and(np.in1d(edges[0], inds),
                              np.in1d(edges[1], inds))
    edges = edges[:, ind_mask]
    if edge_weights is not None:
        weights = edge_weights[ind_mask]
    if len(edges.ravel()):
        maxval = edges.max()
    else:
        maxval = 0
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(maxval + 1))
    edges = order[edges]
    if edge_weights is None:
        return edges
    else:
        return edges, edge_weights


def _to_graph(n_x, n_y, n_z, mask=None, img=None,
              return_as=sparse.coo_matrix, dtype=None):
    """Auxiliary function for img_to_graph and grid_to_graph
    """
    edges, weights = _make_edges(n_x, n_y)

    if dtype is None:
        dtype = img.dtype

    if mask is not None:
        edges, weights = _mask_edges_weights(mask, edges, weights)
        diag = img.squeeze()[mask]
    else:
        diag = img.ravel()
    n_voxels = diag.size

    diag_idx = np.arange(n_voxels)
    i_idx = np.hstack((edges[0], edges[1]))
    j_idx = np.hstack((edges[1], edges[0]))
    graph = sparse.coo_matrix((np.hstack((weights, weights, diag)),
                              (np.hstack((i_idx, diag_idx)),
                               np.hstack((j_idx, diag_idx)))),
                              (n_voxels, n_voxels),
                              dtype=dtype)
    if return_as is np.ndarray:
        return graph.toarray()
    return return_as(graph)


def img_to_graph(img, *, mask=None, return_as=sparse.coo_matrix, dtype=None):
    """
        Graph of the pixel-to-pixel gradient connections

        Edges are weighted with the gradient values.

        Parameters
        ----------
        img : ndarray of shape (height, width) or (height, width, channel)
            2D or 3D image.
        mask : ndarray of shape (height, width) or \
                (height, width, channel), dtype=bool, default=None
            An optional mask of the image, to consider only part of the
            pixels.
        return_as : np.ndarray or a sparse matrix class, \
                default=sparse.coo_matrix
            The class to use to build the returned adjacency matrix.
    """
    img = np.atleast_3d(img)
    n_x, n_y = img.shape
    return _to_graph(n_x, n_y, mask, img, return_as, dtype)


def grid_to_graph(n_x, n_y, n_z=1, *, mask=None, return_as=sparse.coo_matrix,
                  dtype=int):
    """Graph of the pixel-to-pixel connections

    Edges exist if 2 voxels are connected.

    Parameters
    ----------
    n_x : int
        Dimension in x axis
    n_y : int
        Dimension in y axis
    n_z : int, default=1
        Dimension in z axis
    mask : ndarray of shape (n_x, n_y, n_z), dtype=bool, default=None
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, \
            default=sparse.coo_matrix
        The class to use to build the returned adjacency matrix.
    dtype : dtype, default=int
        The data of the returned sparse matrix. By default it is int

    Notes
    -----
    For scikit-learn versions 0.14.1 and prior, return_as=np.ndarray was
    handled by returning a dense np.matrix instance.  Going forward, np.ndarray
    returns an np.ndarray, as expected.

    For compatibility, user code relying on this method should wrap its
    calls in ``np.asarray`` to avoid type issues.
    """
    return _to_graph(n_x, n_y, n_z, mask=mask, return_as=return_as,
                     dtype=dtype)


