import time
import torch
import torch.nn.functional as F


def main():
    repeat = 100
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    query = torch.rand(32, 8, 128, 64, dtype=dtype, device=device)
    key = torch.rand(32, 8, 128, 64, dtype=dtype, device=device)
    value = torch.rand(32, 8, 128, 64, dtype=dtype, device=device)
    scale_factor = 0.125

    ori_time_list = []
    for _ in range(repeat):
        torch.cuda.synchronize(device=device)
        time_start = time.perf_counter()
        # 原始Self-Attention实现
        res = torch.softmax(query @ key.transpose(-2, -1) * scale_factor, dim=-1) @ value
        torch.cuda.synchronize(device=device)
        time_end = time.perf_counter()
        ori_time_list.append(time_end - time_start)

    fa_time_list = []
    for _ in range(repeat):
        torch.cuda.synchronize(device=device)
        time_start = time.perf_counter()
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            # 使用Pytorch官方提供的FA实现
            res_fa = F.scaled_dot_product_attention(query, key, value, scale=scale_factor)
        torch.cuda.synchronize(device=device)
        time_end = time.perf_counter()
        fa_time_list.append(time_end - time_start)

    diff = (res - res_fa).abs().max()
    ratio = [ori_time_list[i] / fa_time_list[i] for i in range(repeat)]
    avg_ratio = sum(ratio[1:]) / len(ratio[1:])
    print(f"max diff: {diff}")
    print(f"avg speed up ratio: {avg_ratio}")


if __name__ == '__main__':
    main()

