import numpy as np
from sklearn.cluster import KMeans  

def extract_blocks(img_gray, block=4):
    h, w = img_gray.shape
    ph = (block - (h % block)) % block
    pw = (block - (w % block)) % block
    img_p = np.pad(img_gray, ((0, ph), (0, pw)), 'constant', constant_values=0)
    H, W = img_p.shape
    blocks = img_p.reshape(H//block, block, W//block, block).swapaxes(1,2).reshape(-1, block, block)
    return blocks, img_gray.shape, (H, W)


def merge_blocks(blocks, orig_shape, padded_shape, block=4):
    H, W = padded_shape
    nb_h = H//block
    nb_w = W//block
    arr = np.array(blocks).reshape(nb_h, nb_w, block, block).swapaxes(1,2).reshape(H, W)
    return arr[:orig_shape[0], :orig_shape[1]]


def train_codebook(blocks, n_codevectors=256, random_state=0):
    """
    blocks: array shape (nblocks, block, block)
    returns codebook shape (K, block*block)
    """
    X = np.array(blocks).reshape(len(blocks), -1).astype(np.float32)
    # Using sklearn KMeans if available - faster and robust
    kmeans = KMeans(n_clusters=n_codevectors, random_state=random_state, n_init=4, max_iter=100)
    kmeans.fit(X)
    codebook = kmeans.cluster_centers_.astype(np.int16)
    return codebook


def encode_vq(blocks, codebook):
    """
    blocks: (nblocks, block, block)
    codebook: (K, dim)
    returns indices (nblocks,) dtype=int
    """
    X = np.array(blocks).reshape(len(blocks), -1).astype(np.float32)
    # compute distances to codebook
    # brute-force distances: (nblocks, K)
    dists = np.sum((X[:, None, :] - codebook[None, :, :].astype(np.float32))**2, axis=2)
    indices = np.argmin(dists, axis=1).astype(np.int32)
    return indices


def decode_vq(indices, codebook, orig_shape, padded_shape, block=4):
    X = codebook[indices]  # (nblocks, dim)
    blocks = X.reshape(-1, block, block).astype(np.uint8)
    recon = merge_blocks(blocks, orig_shape, padded_shape, block=block)
    return recon