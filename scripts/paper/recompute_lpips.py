"""Recompute LPIPS (AlexNet) and a PSNR spot check from the tensors that
scripts/inference_wm_one_stage.py saves (pred_videos.pt, gt_videos.pt), in fixed-size chunks.
Used for the WM2 oracle evaluation on the OOD categories.

Example:
    python scripts/paper/recompute_lpips.py \
        --run no_cube=outputs/wm2_oracle_no_cube --run cube=outputs/wm2_oracle_cube
"""

import argparse

import lpips
import torch


def psnr_of(pred01, gt01):
    mse = ((pred01 - gt01) ** 2).mean(dim=[1, 2, 3])
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR",
                        help="Output folder of an inference_wm_one_stage.py run; repeat per run")
    parser.add_argument("--num_history", type=int, default=5, help="History frames at the start of gt_videos.pt")
    parser.add_argument("--chunk_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    results = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        pred_videos = torch.load(f"{path}/pred_videos.pt", map_location="cpu")
        gt_videos = torch.load(f"{path}/gt_videos.pt", map_location="cpu")
        gt_future = gt_videos[:, args.num_history:]

        B, T = pred_videos.shape[0], pred_videos.shape[1]
        assert gt_future.shape[:2] == (B, T), (gt_future.shape, pred_videos.shape)

        pred_flat = (pred_videos / 2.0 + 0.5).clamp(0, 1).reshape(B * T, *pred_videos.shape[2:]).float()
        gt_flat = (gt_future / 2.0 + 0.5).clamp(0, 1).reshape(B * T, *gt_future.shape[2:]).float()

        lpips_chunks, psnr_sum = [], 0.0
        with torch.no_grad():
            for start in range(0, pred_flat.shape[0], args.chunk_size):
                p = pred_flat[start:start + args.chunk_size].to(device)
                g = gt_flat[start:start + args.chunk_size].to(device)
                lpips_chunks.append(lpips_fn(p, g, normalize=True).squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu())
                psnr_sum += psnr_of(p, g) * p.shape[0]

        lpips_avg = float(torch.cat(lpips_chunks, dim=0).reshape(B, T).numpy().mean(axis=1).mean())
        psnr_spotcheck = psnr_sum / pred_flat.shape[0]
        results[name] = (lpips_avg, psnr_spotcheck, B)
        print(f"{name}: n_sequences={B}  lpips_avg={lpips_avg:.6f}  psnr_spotcheck={psnr_spotcheck:.4f}", flush=True)

    print("\n=== Summary ===")
    for name, (lp, ps, b) in results.items():
        print(f"{name:15s} n={b:4d}  LPIPS={lp:.4f}  PSNR(spotcheck)={ps:.2f}")


if __name__ == "__main__":
    main()
