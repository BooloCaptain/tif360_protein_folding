import torch
import torch.nn.functional as F

def test_flash_attention():
    # FlashAttention strictly requires float16 or bfloat16!
    dtype = torch.bfloat16 
    device = torch.device("cuda")

    # B=Batch, nhead=Heads, L=SeqLen, head_dim=Dimension per head
    # Note: head_dim MUST be a multiple of 8 (e.g., 64, 128)
    q = torch.randn(8, 8, 4096, 64, dtype=dtype, device=device)
    k = torch.randn(8, 8, 4096, 64, dtype=dtype, device=device)
    v = torch.randn(8, 8, 4096, 64, dtype=dtype, device=device)

    print("Attempting forced FlashAttention...")
    try:
        # Force PyTorch to ONLY use FlashAttention
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            out = F.scaled_dot_product_attention(q, k, v)
        print("[SUCCESS] FlashAttention is perfectly enabled on your hardware!")
    except Exception as e:
        print("[FAILED] FlashAttention could not run. PyTorch threw this error:")
        print(e)

if __name__ == "__main__":
    test_flash_attention()