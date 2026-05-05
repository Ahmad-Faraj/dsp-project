import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq
from collections import Counter
import concurrent.futures

# ───────────────────────────────────────────
# 1. FAST DCT HELPERS (Using OpenCV C++)
# ───────────────────────────────────────────
def dct2(block):
    return cv2.dct(block.astype(np.float32))

def idct2(block):
    return cv2.idct(block.astype(np.float32))

# ───────────────────────────────────────────
# 2. JPEG QUANTIZATION MATRIX & ZIGZAG
# ───────────────────────────────────────────
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

_ZIGZAG_IDX = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]

def zigzag(block):
    return [block[r, c] for r, c in _ZIGZAG_IDX]

def rle(data):
    out = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            out.append((data[i - 1], count))
            count = 1
    out.append((data[-1], count))
    return out

# ───────────────────────────────────────────
# 3. ENTROPY CODING (HUFFMAN)
# ───────────────────────────────────────────
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freq = Counter(data)
    heap = [HuffmanNode(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)
    
    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = HuffmanNode(None, node.freq)
        root.left = node
        return root
        
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
        
    return heap[0] if heap else None

def build_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        build_codes(node.left, prefix + "0", codebook)
        build_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(data_list):
    if not data_list:
        return bytes()
    symbols = [str(item) for item in data_list]
    root = build_huffman_tree(symbols)
    codebook = build_codes(root)
    bitstring = "".join(codebook[sym] for sym in symbols)
    pad_len = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * pad_len
    byte_arr = bytearray()
    for i in range(0, len(bitstring), 8):
        byte_arr.append(int(bitstring[i:i+8], 2))
    return bytes(byte_arr)

# ───────────────────────────────────────────
# 4. I-FRAME COMPRESSION
# ───────────────────────────────────────────
def encode_iframe(frame_channel):
    h, w = frame_channel.shape
    comp = np.zeros((h, w), dtype=np.float32)
    stream = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = frame_channel[i:i+8, j:j+8].astype(np.float32) - 128
            q = np.round(dct2(block) / Q)
            comp[i:i+8, j:j+8] = q
            stream += rle(zigzag(q))
    return comp, stream

def decode_iframe(comp):
    h, w = comp.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = comp[i:i+8, j:j+8] * Q
            out[i:i+8, j:j+8] = idct2(block) + 128
    return np.clip(out, 0, 255).astype(np.uint8)

# ───────────────────────────────────────────
# 5. P-FRAME COMPRESSION (FAST THREE-STEP SEARCH)
# ───────────────────────────────────────────
def motion_est_fast(curr, ref, bs=16, search_area=8):
    h, w = curr.shape
    residual = np.zeros_like(curr, dtype=np.int16)
    vectors = []
    
    for i in range(0, h, bs):
        for j in range(0, w, bs):
            step = max(1, search_area // 2)
            cx, cy = j, i
            
            while step >= 1:
                best_diff = float('inf')
                best_dx, best_dy = 0, 0
                
                # Check 9 points around current center
                for dy in [-step, 0, step]:
                    for dx in [-step, 0, step]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny <= h - bs and 0 <= nx <= w - bs:
                            diff = np.sum(np.abs(curr[i:i+bs, j:j+bs].astype(np.int32) - ref[ny:ny+bs, nx:nx+bs].astype(np.int32)))
                            if diff < best_diff:
                                best_diff = diff
                                best_dx, best_dy = dx, dy
                
                cx += best_dx
                cy += best_dy
                
                if step == 1:
                    break
                step //= 2
                
            final_dy = cy - i
            final_dx = cx - j
            vectors.append((final_dy, final_dx))
            residual[i:i+bs, j:j+bs] = curr[i:i+bs, j:j+bs].astype(np.int16) - ref[cy:cy+bs, cx:cx+bs].astype(np.int16)
            
    return vectors, residual

def encode_residual(residual):
    h, w = residual.shape
    comp = np.zeros((h, w), dtype=np.float32)
    stream = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = residual[i:i+8, j:j+8].astype(np.float32) 
            q = np.round(dct2(block) / (Q * 1.5)) 
            comp[i:i+8, j:j+8] = q
            stream += rle(zigzag(q))
    return comp, stream

def decode_residual(comp):
    h, w = comp.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = comp[i:i+8, j:j+8] * (Q * 1.5)
            out[i:i+8, j:j+8] = idct2(block)
    return out

def decode_pframe(ref_recon, vectors, comp_residual, bs=16):
    h, w = comp_residual.shape
    decoded_res = decode_residual(comp_residual)
    recon = np.zeros((h, w), dtype=np.float32)
    v_idx = 0
    for i in range(0, h, bs):
        for j in range(0, w, bs):
            dy, dx = vectors[v_idx]
            v_idx += 1
            recon[i:i+bs, j:j+bs] = ref_recon[i+dy:i+dy+bs, j+dx:j+dx+bs].astype(np.float32) + decoded_res[i:i+bs, j:j+bs]
    return np.clip(recon, 0, 255).astype(np.uint8)

# ───────────────────────────────────────────
# 6. BITSTREAM FORMATION
# ───────────────────────────────────────────
FRAME_TYPE_I = 0
FRAME_TYPE_P = 1

def pack_frame_huffman(frame_idx, frame_type, data_list):
    encoded_bytes = huffman_encode(data_list)
    buf = bytearray()
    buf += frame_idx.to_bytes(4, 'big')
    buf += frame_type.to_bytes(1, 'big')
    buf += len(encoded_bytes).to_bytes(4, 'big')
    buf += encoded_bytes
    return buf

def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return np.inf
    return 10 * np.log10(255 ** 2 / mse)

# ───────────────────────────────────────────
# 7. MAIN PIPELINE (FAST 4:2:0 YUV)
# ───────────────────────────────────────────
def run_pipeline(params, log, progress, done):
    cap = cv2.VideoCapture(params["video_path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    reference_channels = [None, None, None] 
    frame_types = []
    psnr_values = []
    frame_pairs = []
    
    original_size = 0
    compressed_size = 0

    max_frames  = params["max_frames"]
    i_interval  = params["i_interval"]
    block_size  = params["block_size"]
    search_area = params["search_area"]
    output_dir  = params.get("output_dir", "outputs")

    os.makedirs(output_dir, exist_ok=True)
    bitstream_path = os.path.join(output_dir, "video.bin")
    recon_video_path = os.path.join(output_dir, "reconstructed.avi")
    
    out_video = None
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    def process_pframe_channel(curr_c, ref_c, bs, area):
        vectors, residual = motion_est_fast(curr_c, ref_c, bs=bs, search_area=area)
        comp_res, stream_res = encode_residual(residual)
        recon_c = decode_pframe(ref_c, vectors, comp_res, bs=bs)
        return vectors, stream_res, recon_c

    with open(bitstream_path, "wb") as bs_file:
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            h, w, _ = yuv.shape
            
            # Crop to be divisible by block_size * 2 (for safe 4:2:0 subsampling)
            h_adj = (h // (block_size * 2)) * (block_size * 2)
            w_adj = (w // (block_size * 2)) * (block_size * 2)
            yuv_cropped = yuv[:h_adj, :w_adj, :]

            # CHROMA SUBSAMPLING (4:2:0) -> Y is full size, U/V are half size
            Y = yuv_cropped[:, :, 0]
            U = cv2.resize(yuv_cropped[:, :, 1], (w_adj // 2, h_adj // 2), interpolation=cv2.INTER_AREA)
            V = cv2.resize(yuv_cropped[:, :, 2], (w_adj // 2, h_adj // 2), interpolation=cv2.INTER_AREA)
            
            curr_channels = [Y, U, V]
            recon_channels = [None, None, None]
            
            # Block sizes adapt to the smaller U and V channels
            bs_list = [block_size, max(8, block_size // 2), max(8, block_size // 2)]
            area_list = [search_area, max(2, search_area // 2), max(2, search_area // 2)]

            original_size += Y.nbytes + U.nbytes + V.nbytes
            
            if out_video is None:
                out_video = cv2.VideoWriter(recon_video_path, fourcc, fps, (w_adj, h_adj))

            if i % i_interval == 0:
                log(f"Encoding I-frame {i} (Fast DCT + 4:2:0)")
                huff_data_all = []
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(encode_iframe, curr_channels[c]) for c in range(3)]
                    for c, future in enumerate(futures):
                        comp, stream = future.result()
                        recon_channels[c] = decode_iframe(comp)
                        huff_data_all.extend([f"C{c}_{v}_{cnt}" for v, cnt in stream])

                packet = pack_frame_huffman(i, FRAME_TYPE_I, huff_data_all)
                bs_file.write(packet)
                compressed_size += len(packet)

                reference_channels = [rc.copy() for rc in recon_channels]
                frame_types.append("I")

            else:
                log(f"Encoding P-frame {i} (Three-Step Search)")
                huff_data_all = []
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(process_pframe_channel, curr_channels[c], reference_channels[c], bs_list[c], area_list[c]) for c in range(3)]
                    for c, future in enumerate(futures):
                        vectors, stream_res, recon = future.result()
                        recon_channels[c] = recon
                        huff_data_all.extend([f"MVC{c}_{dy}_{dx}" for dy, dx in vectors])
                        huff_data_all.extend([f"C{c}_{v}_{cnt}" for v, cnt in stream_res])

                packet = pack_frame_huffman(i, FRAME_TYPE_P, huff_data_all)
                bs_file.write(packet)
                compressed_size += len(packet)

                reference_channels = [rc.copy() for rc in recon_channels]
                frame_types.append("P")

            # Upsample U and V back to full resolution for viewing
            U_up = cv2.resize(recon_channels[1], (w_adj, h_adj), interpolation=cv2.INTER_LINEAR)
            V_up = cv2.resize(recon_channels[2], (w_adj, h_adj), interpolation=cv2.INTER_LINEAR)
            recon_yuv_full = np.dstack((recon_channels[0], U_up, V_up)).astype(np.uint8)

            psnr_values.append(psnr(yuv_cropped, recon_yuv_full))
            
            recon_bgr = cv2.cvtColor(recon_yuv_full, cv2.COLOR_YUV2BGR)
            out_video.write(recon_bgr)

            if len(frame_pairs) < 6:
                orig_rgb = cv2.cvtColor(frame[:h_adj, :w_adj, :], cv2.COLOR_BGR2RGB)
                recon_rgb = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB)
                frame_pairs.append((orig_rgb, recon_rgb))

            progress(int(100 * (i + 1) / max_frames))

    cap.release()
    if out_video:
        out_video.release()
        
    log("Saved: video.bin")
    log("Saved: reconstructed.avi")

    avg_psnr = np.mean(psnr_values)
    ratio = round(original_size / compressed_size, 2) if compressed_size != 0 else 1

    log("Saving PSNR chart...")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    colors = ["#E53935" if t == "I" else "#1E88E5" for t in frame_types]
    ax1.bar(range(len(psnr_values)), psnr_values, color=colors, width=0.8)
    ax1.plot(range(len(psnr_values)), psnr_values, color="#333333", linewidth=1)
    ax1.axhline(avg_psnr, color="#FF6F00", linestyle="--", linewidth=1.5, label=f"Avg: {avg_psnr:.2f} dB")
    ax1.set_title("PSNR Per Frame (Fast Mode 4:2:0)")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("PSNR (dB)")
    ax1.legend()
    fig1.tight_layout()
    psnr_path = os.path.join(output_dir, "psnr_chart.png")
    fig1.savefig(psnr_path, dpi=150)
    plt.close(fig1)

    log("Saving frame type chart...")
    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
    frame_values = [1 if f == "I" else 0 for f in frame_types]
    ax2.step(range(len(frame_values)), frame_values, where="mid", color="#333333")
    ax2.fill_between(range(len(frame_values)), frame_values, step="mid", alpha=0.3, color="#E53935")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["P-frame", "I-frame"])
    ax2.set_xlabel("Frame index")
    ax2.set_title("Frame Type Sequence")
    fig2.tight_layout()
    frame_type_path = os.path.join(output_dir, "frame_types.png")
    fig2.savefig(frame_type_path, dpi=150)
    plt.close(fig2)

    log("Saving frame comparison...")
    n = len(frame_pairs)
    if n > 0:
        fig3, axes = plt.subplots(2, n, figsize=(3 * n, 6))
        if n == 1:
            axes = np.expand_dims(axes, axis=1)
        for col, (orig, recon) in enumerate(frame_pairs):
            axes[0, col].imshow(orig)
            axes[0, col].set_title(f"Original #{col}", fontsize=8)
            axes[0, col].axis("off")
            axes[1, col].imshow(recon)
            axes[1, col].set_title(f"Recon #{col}", fontsize=8)
            axes[1, col].axis("off")
        fig3.suptitle("Original vs Reconstructed (Fast 4:2:0)")
        fig3.tight_layout()
        compare_path = os.path.join(output_dir, "frame_comparison.png")
        fig3.savefig(compare_path, dpi=150)
        plt.close(fig3)

    data = {
        "frame_types": frame_types,
        "psnr_values": psnr_values,
        "frame_pairs": frame_pairs,
        "bitstream_path": bitstream_path,
        "recon_video_path": recon_video_path,
    }
    summary = {
        "avg_psnr": avg_psnr,
        "ratio": ratio,
        "total_frames": len(frame_types),
    }

    log(f"Bitstream written → {bitstream_path}")
    log(f"Compression Ratio -> {ratio}:1")
    log("Video compression completed.")
    done(data, summary)