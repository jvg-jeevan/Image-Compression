import numpy as np
import cv2
from bitarray import bitarray

JPEG_STD_Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)


def block_split(img, block=8):
    h, w = img.shape
    # pad to multiple of block
    ph = (block - (h % block)) % block
    pw = (block - (w % block)) % block
    img_p = np.pad(img, ((0, ph), (0, pw)), 'constant', constant_values=0)
    H, W = img_p.shape
    blocks = img_p.reshape(H//block, block, W//block, block).swapaxes(1,2).reshape(-1, block, block)
    return blocks, img.shape, (H, W)


def block_merge(blocks, orig_shape, padded_shape, block=8):
    H, W = padded_shape
    nb_h = H//block
    nb_w = W//block
    arr = np.array(blocks).reshape(nb_h, nb_w, block, block).swapaxes(1,2).reshape(H, W)
    return arr[:orig_shape[0], :orig_shape[1]]


def block_dct(block):
    return cv2.dct(block.astype(np.float32))


def block_idct(block):
    return cv2.idct(block.astype(np.float32))


def quantize_block(dct_block, qtable):
    return np.round(dct_block / qtable).astype(np.int32)


def dequantize_block(q_block, qtable):
    return (q_block.astype(np.float32) * qtable)


def encode_dct_image(img_gray, qscale=1.0, qtable=None):
    """
    Returns:
      quantized_blocks_flat: 1D numpy array of quantized coefficient integers (flattened)
      meta: dict containing shapes and parameters required for decoding
    """
    if qtable is None:
        qtable = JPEG_STD_Q * qscale
    blocks, orig_shape, padded_shape = block_split(img_gray, block=8)
    qblocks = []
    for b in blocks:
        d = block_dct(b - 128.0)    # shift to signed range like JPEG
        q = quantize_block(d, qtable)
        qblocks.append(q)
    qblocks = np.array(qblocks)  # shape (nblocks,8,8)
    flat = qblocks.flatten()
    meta = {"orig_shape": orig_shape, "padded_shape": padded_shape, "qscale": qscale, "qtable": qtable}
    return flat.astype(np.int32), meta


def decode_dct_image(flat_qcoeffs, meta):
    qtable = meta["qtable"]
    orig_shape = meta["orig_shape"]
    padded_shape = meta["padded_shape"]
    block = 8
    nblocks = flat_qcoeffs.size // (block*block)
    qblocks = flat_qcoeffs.reshape(nblocks, block, block)
    recon_blocks = []
    for qb in qblocks:
        d = dequantize_block(qb, qtable)
        b = block_idct(d) + 128.0
        b = np.clip(np.round(b), 0, 255).astype(np.uint8)
        recon_blocks.append(b)
    recon = block_merge(recon_blocks, orig_shape, padded_shape, block=8)
    return recon


def rebase_to_unsigned(arr):
    mx = np.max(arr)
    mn = np.min(arr)
    offset = -mn if mn < 0 else 0
    rebased = arr + offset
    return rebased.astype(np.int64), int(offset)


def rebase_back_unsigned(arr, offset):
    return (arr - offset).astype(np.int32)