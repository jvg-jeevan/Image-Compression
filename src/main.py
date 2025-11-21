import io
import time
from math import inf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

from huffman import build_huffman_tree, generate_codes, huffman_encode, huffman_decode
from arithmetic import ArithmeticCoder
from rice import rice_encode, rice_decode
from dct import encode_dct_image, decode_dct_image, rebase_to_unsigned, rebase_back_unsigned
from vq import extract_blocks, train_codebook, encode_vq, decode_vq

DEFAULT_IMAGE_PATH = None

def to_gray(img):
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.uint8)


def entropy(prob):
    return -np.sum(prob * np.log2(prob + 1e-12))


def prepare_cumfreq(px):
    values, counts = np.unique(px, return_counts=True)
    freq = np.zeros(256, dtype=int)
    freq[values] = counts
    cum = [0]
    r = 0
    for f in freq:
        r += int(f)
        cum.append(r)
    return cum


def plot_grouped_comparison(labels, bpp_vals, cr_vals, eff_vals):
    n = len(labels)
    x = np.arange(n)
    width = 0.25

    bpp = np.array(bpp_vals)
    cr = np.array(cr_vals)
    eff = np.array(eff_vals)

    max_bpp = bpp.max() if bpp.size > 0 else 1
    max_cr = cr.max() if cr.size > 0 else 1
    max_eff = eff.max() if eff.size > 0 else 1

    cr_scaled = cr / max_cr * max_bpp
    eff_scaled = eff / max_eff * max_bpp

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width, bpp, width, label='BPP')
    ax.bar(x, cr_scaled, width, label='CR (scaled)')
    ax.bar(x + width, eff_scaled, width, label='Efficiency (scaled)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Comparison: BPP | CR | Efficiency")
    ax.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def to_rgb_uint8(img_gray):
    if img_gray is None:
        return None
    if img_gray.ndim == 2:
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return img_gray


def run_huffman_on_bytes(px_bytes):
    vals, counts = np.unique(px_bytes, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    root = build_huffman_tree(freq)
    codes = generate_codes(root)
    enc = huffman_encode(px_bytes.tolist(), codes)
    dec = np.array(huffman_decode(enc, root), dtype=np.uint8)
    return len(enc), dec


def run_arithmetic(px_bytes):
    cum = prepare_cumfreq(px_bytes)
    coder = ArithmeticCoder()
    enc = coder.encode(px_bytes, cum)
    dec = np.array(coder.decode(enc, len(px_bytes), cum), dtype=np.uint8)
    return len(enc), dec


def run_rice(px_bytes):
    best_k = 0
    best_bits = inf
    best_enc = None

    for k in range(0, 9):
        enc = rice_encode(px_bytes, k)
        bits = len(enc)
        if bits < best_bits:
            best_bits = bits
            best_k = k
            best_enc = enc

    dec = np.array(rice_decode(best_enc, best_k, len(px_bytes)), dtype=np.uint8)
    return best_bits, best_k, dec


def run_huffman_on_array(arr):
    vals, counts = np.unique(arr, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    root = build_huffman_tree(freq)
    codes = generate_codes(root)
    enc = huffman_encode(arr.tolist(), codes)
    dec = np.array(huffman_decode(enc, root), dtype=np.int32)
    return len(enc), dec


def method_huffman(gray):
    px = gray.flatten()
    bits, recon = run_huffman_on_bytes(px)
    return bits, recon.reshape(gray.shape)


def method_arithmetic(gray):
    px = gray.flatten()
    bits, recon = run_arithmetic(px)
    return bits, recon.reshape(gray.shape)


def method_rice(gray):
    px = gray.flatten()
    bits, best_k, recon = run_rice(px)
    return bits, best_k, recon.reshape(gray.shape)


def method_dct(gray):
    flat_q, meta = encode_dct_image(gray, qscale=1.0)
    rebased, offset = rebase_to_unsigned(flat_q)
    bits, dec_rebased = run_huffman_on_array(rebased)
    dec_flat = rebase_back_unsigned(dec_rebased, offset)
    recon = decode_dct_image(dec_flat, meta)
    return bits, offset, recon


def method_vq(gray):
    block = 4
    K = 256
    blocks, orig_shape, padded_shape = extract_blocks(gray, block=block)

    K_actual = min(K, max(2, len(blocks)))
    codebook = train_codebook(blocks, n_codevectors=K_actual)
    indices = encode_vq(blocks, codebook)

    bits, dec_indices = run_huffman_on_array(indices)
    recon = decode_vq(dec_indices.astype(np.int32), codebook, orig_shape, padded_shape, block)

    return bits, K_actual, recon


def compress_and_show_all(img):
    if img is None:
        return "No image", None, None, None, None, None, None

    gray = to_gray(img)
    h, w = gray.shape
    N = h * w
    orig_bits = N * 8

    values, counts = np.unique(gray.flatten(), return_counts=True)
    ent = entropy(counts / counts.sum())

    summary = []

    labels = []
    bpp_vals = []
    cr_vals = []
    eff_vals = []

    t0 = time.time()
    bits_h, recon_h = method_huffman(gray)
    t1 = time.time()

    bpp = bits_h / N
    cr = orig_bits / bits_h
    eff = ent / bpp

    summary.append(
        f"Huffman:\n"
        f"    bits = {bits_h}\n"
        f"    bpp = {bpp:.4f}\n"
        f"    CR = {cr:.3f}\n"
        f"    Efficiency = {eff:.3f}\n"
        f"    time = {t1 - t0:.2f}s\n"
    )
    labels.append("Huffman")
    bpp_vals.append(bpp)
    cr_vals.append(cr)
    eff_vals.append(eff)

    t0 = time.time()
    bits_a, recon_a = method_arithmetic(gray)
    t1 = time.time()

    bpp = bits_a / N
    cr = orig_bits / bits_a
    eff = ent / bpp

    summary.append(
        f"Arithmetic:\n"
        f"    bits = {bits_a}\n"
        f"    bpp = {bpp:.4f}\n"
        f"    CR = {cr:.3f}\n"
        f"    Efficiency = {eff:.3f}\n"
        f"    time = {t1 - t0:.2f}s\n"
    )
    labels.append("Arithmetic")
    bpp_vals.append(bpp)
    cr_vals.append(cr)
    eff_vals.append(eff)

    t0 = time.time()
    bits_r, best_k, recon_r = method_rice(gray)
    t1 = time.time()

    bpp = bits_r / N
    cr = orig_bits / bits_r
    eff = ent / bpp

    summary.append(
        f"Rice (k={best_k}):\n"
        f"    bits = {bits_r}\n"
        f"    bpp = {bpp:.4f}\n"
        f"    CR = {cr:.3f}\n"
        f"    Efficiency = {eff:.3f}\n"
        f"    time = {t1 - t0:.2f}s\n"
    )
    labels.append("Rice")
    bpp_vals.append(bpp)
    cr_vals.append(cr)
    eff_vals.append(eff)

    t0 = time.time()
    bits_d, offset_d, recon_d = method_dct(gray)
    t1 = time.time()

    bpp = bits_d / N
    cr = orig_bits / bits_d
    eff = ent / bpp

    summary.append(
        f"DCT:\n"
        f"    bits = {bits_d}\n"
        f"    bpp = {bpp:.4f}\n"
        f"    CR = {cr:.3f}\n"
        f"    Efficiency = {eff:.3f}\n"
        f"    time = {t1 - t0:.2f}s\n"
    )
    labels.append("DCT")
    bpp_vals.append(bpp)
    cr_vals.append(cr)
    eff_vals.append(eff)

    t0 = time.time()
    bits_v, K_actual, recon_v = method_vq(gray)
    t1 = time.time()

    bpp = bits_v / N
    cr = orig_bits / bits_v
    eff = ent / bpp

    summary.append(
        f"VQ (block=4, K={K_actual}):\n"
        f"    bits = {bits_v}\n"
        f"    bpp = {bpp:.4f}\n"
        f"    CR = {cr:.3f}\n"
        f"    Efficiency = {eff:.3f}\n"
        f"    time = {t1 - t0:.2f}s\n"
    )
    labels.append("VQ")
    bpp_vals.append(bpp)
    cr_vals.append(cr)
    eff_vals.append(eff)

    comp_plot = plot_grouped_comparison(labels, bpp_vals, cr_vals, eff_vals)

    summary_text = f"Entropy: {ent:.4f}\n\n" + "\n".join(summary)

    return (
        summary_text,
        to_rgb_uint8(recon_h),
        to_rgb_uint8(recon_a),
        to_rgb_uint8(recon_r),
        to_rgb_uint8(recon_d),
        to_rgb_uint8(recon_v),
        comp_plot
    )


with gr.Blocks() as demo:
    gr.Markdown("## Image Compression (Huffman • Arithmetic • Rice • DCT • VQ)")

    with gr.Row():
        img_in = gr.Image(type="numpy", label="Upload Image", value=DEFAULT_IMAGE_PATH)
        compress_btn = gr.Button("Compress")

    with gr.Row():
        out_huff = gr.Image(label="Huffman (reconstructed)")
        out_arith = gr.Image(label="Arithmetic (reconstructed)")
        out_rice = gr.Image(label="Rice (reconstructed)")

    with gr.Row():
        out_dct = gr.Image(label="DCT (reconstructed)")
        out_vq = gr.Image(label="VQ (reconstructed)")

    with gr.Row():
        out_text = gr.Textbox(
            label="Summary (bits | bpp | CR | Efficiency | time)",
            lines=18
        )

    with gr.Row():
        comp_plot = gr.Image(
            label="Comparison Plot (BPP | CR | Efficiency)"
        )

    compress_btn.click(
        compress_and_show_all,
        inputs=[img_in],
        outputs=[out_text, out_huff, out_arith, out_rice, out_dct, out_vq, comp_plot]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)