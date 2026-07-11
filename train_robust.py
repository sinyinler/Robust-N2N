# -*- coding: utf-8 -*-
"""Robust-N2N 一步法训练入口。

数据与配对完全复用 train_n2n 的 build_loaders（同场景 Δ∈{5,7,9} 的噪声对 n1/n2，
支持 /mnt2/songyd/mix 的 npy 与 lbf）。区别：
  - 网络：RobustDenoiser（轻量 U-Net + 3 处 GIBlock，训练注入/推理关闭）；
  - 每个 batch 对 n1、n2 都前向，得到 f(n1)、f(n2)；
  - 损失：RobustN2NLoss = 对称N2N(Charbonnier) + 一致性 + [过半后]残差白度 + RTV；
  - 白度项在训练进度过 whiten_start_frac 后启用。
"""
from __future__ import annotations

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 复用 N2N 的数据加载 / 调度 / 随机种子（数据与配对口径完全一致）
from train_n2n import build_loaders, set_seed, build_onecycle
from models.denoiser_feats import DenoiserWithFeats, FEAT_CHANNELS
from models.aux_decoder import AuxDecoder
from losses.robust_n2n_loss import RobustN2NLoss
from losses.feature_consistency import FeatureConsistencyLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robust-N2N one-step training (LiteN2N U-Net + GIBlock + combined loss).")
    # ---- 数据（与 train_n2n 一致，供 build_loaders 使用）----
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--data_subdir", type=str, default="npy")
    p.add_argument("--strict_data_subdir", type=int, default=1)
    p.add_argument("--data_index_min", type=int, default=-1)
    p.add_argument("--data_index_max", type=int, default=-1)
    p.add_argument("--levels", type=int, nargs="*", default=None)
    p.add_argument("--mix_root", type=str, default="")
    p.add_argument("--mix_scenes", type=str, nargs="*", default=None)
    p.add_argument("--mix_subdirs", type=str, nargs="*", default=None)
    p.add_argument("--intervals", type=int, nargs="*", default=[5, 7, 9])
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=24, help="对 n1、n2 各前向一次，显存约 2×，故默认比 N2N 小。")
    p.add_argument("--max_pixels_per_batch", type=int, default=0)
    p.add_argument("--batch_ref_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--train_fraction", type=float, default=0.99)
    p.add_argument("--val_limit_batches", type=int, default=20)
    p.add_argument("--intensity_transform", choices=["log1p", "boxcox", "learned_vst"], default="log1p")
    p.add_argument("--vst_lut", type=str, default="")
    p.add_argument("--boxcox_lam", type=float, default=-0.15)
    p.add_argument("--boxcox_eps", type=float, default=1e-6)
    p.add_argument("--lambda_conditioned", type=int, default=0)
    p.add_argument("--lambda_min", type=float, default=-0.3)
    p.add_argument("--lambda_max", type=float, default=0.2)
    p.add_argument("--lambda_candidates", type=float, nargs="*",
                   default=[-0.3, -0.25, -0.2, -0.15, -0.1, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2])
    # ---- 训练调度 ----
    p.add_argument("--save_dir", type=str, default="results/checkpoints/robust_n2n")
    p.add_argument("--log_dir", type=str, default="results/logs/robust_n2n")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01, help="alias for lr_max")
    p.add_argument("--lr_max", type=float, default=None)
    p.add_argument("--lr_final", type=float, default=0.0005)
    p.add_argument("--warmup_pct", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--data_parallel", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    # ---- 组合损失 / GIBlock ----
    p.add_argument("--alpha", type=float, default=1.0, help="N2N 正向 Charbonnier 权重")
    p.add_argument("--beta", type=float, default=1.0, help="N2N 反向 Charbonnier 权重")
    p.add_argument("--gamma", type=float, default=0.1, help="一致性 |f(n1)-f(n2)| 权重")
    p.add_argument("--pixel_target", choices=["n2n", "self"], default="n2n",
                   help="像素靶：n2n=对称N2N(拿兄弟帧当靶)；self=自重建 f(n1)->n1(恒等靶,对照实验)")
    p.add_argument("--w_white", type=float, default=0.05, help="残差白度总权重（小）")
    p.add_argument("--beta_freq", type=float, default=2e-3, help="白度内部 freq 相对 spatial 的权重")
    p.add_argument("--rtv_weight", type=float, default=0.01, help="RTV 权重（建议对齐原 N2N 训练）")
    p.add_argument("--highpass_ratio", type=float, default=0.0)
    p.add_argument("--whiten_start_frac", type=float, default=0.5, help="训练进度过此比例后才开白度项")
    p.add_argument("--inject_sigma", type=float, default=1.0, help="[已弃用] GIBlock 注入，本分支不用")
    p.add_argument("--init_noise_scale", type=float, default=0.1, help="[已弃用] GIBlock 注入，本分支不用")
    # ---- 跨视图特征一致性（SimSiam 式，深层 out3+bridge）----
    p.add_argument("--w_feat", type=float, default=0.1, help="跨视图特征一致性总权重")
    p.add_argument("--feat_dim", type=int, default=128, help="projector 投影维度；0=各尺度用原生通道(C→C)")
    p.add_argument("--feat_pred_hidden", type=int, default=None,
                   help="predictor bottleneck 维度；不给则自动取 dim//4（SimSiam 论文推荐）")
    p.add_argument("--feat_use_proj", type=int, default=1, help="1=带 projector；0=直接用编码器特征当 z")
    p.add_argument("--feat_scales", type=str, nargs="*", default=["encoder3", "bottleneck"],
                   choices=["encoder1", "encoder2", "encoder3", "out3", "bottleneck", "bridge"],
                   help="在哪些深度加 projector 算一致性（浅→深）。out3=encoder3, bridge=bottleneck 为别名")
    p.add_argument("--feat_weights", type=float, nargs="*", default=None,
                   help="各尺度权重，长度须与 --feat_scales 一致；不给则用默认(enc1:0.1,enc2:0.2,enc3:0.5,bn:1.0)")
    p.add_argument("--feat_normalize", type=int, default=0,
                   help="1=尺度权重归一化为和=1（w_feat 成为唯一总强度旋钮，权重变成分布）")
    p.add_argument("--feat_pool", type=int, default=0,
                   help="投影前把特征自适应平均池化到 G×G（0=不池化，逐像素余弦）。"
                        "G=64 可拉平各尺度显存（否则 e1 的 128 维 projector 约 38GB），"
                        "并把一致性从像素级变为区域级，不再强制高频不变。")
    p.add_argument("--feat_pred_constant_lr", type=float, default=0.0,
                   help=">0 时 predictor 用独立优化器、恒定 lr（不参与 OneCycle 衰减）。"
                        "SimSiam §4.2 Table 1c：predictor 不衰减 lr 结果更好，论文正文即采用此设置。")
    p.add_argument("--save_feat_head", type=int, default=0,
                   help="1=同时保存 projector/predictor（及 aux 解码器）权重，供事后分析")
    # ---- 瓶颈辅助重建：给 bridge 一个「必须携带信息」的理由（训练专用，不进 checkpoint）----
    p.add_argument("--w_aux", type=float, default=0.0,
                   help=">0 时启用无 skip 辅助解码器：w_aux·Charb(AuxDec(bridge(n1)), n2)。"
                        "诊断显示 bridge 只承载约 10% 的场景身份信息，纯不变性目标可靠丢信息满足；"
                        "该项强制 bridge 携带可重建整幅图的结构。")
    p.add_argument("--bias_free", type=int, default=0,
                   help="1=Bias-Free 改造（去 conv bias / BN→BFBatchNorm / ELA→Identity），"
                        "使 log 域网络一阶齐次，跨噪声等级泛化（ICLR 2020）。log1p/expm1 保留。")
    args = p.parse_args()
    if args.data_index_min < 0:
        args.data_index_min = None
    if args.data_index_max < 0:
        args.data_index_max = None
    if args.lr_max is None:
        args.lr_max = args.lr
    if args.lr_max <= args.lr_final:
        raise ValueError(f"lr_max({args.lr_max}) must be > lr_final({args.lr_final})")
    return args


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.save_dir, exist_ok=True)

    _, train_loader, val_loader = build_loaders(args)
    model = DenoiserWithFeats(input_channels=1).to(device)   # 无 GIBlock
    if args.bias_free:
        from models.bias_free import make_bias_free, count_additive_constants
        make_bias_free(model)
        model = model.to(device)     # 新建的 BFBatchNorm2d 默认在 CPU，须再搬回 device
        print(f"[INFO] Bias-Free 改造完成，残留加性成分={count_additive_constants(model)}（应全 0）")
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    criterion = RobustN2NLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                              w_white=args.w_white, beta_freq=args.beta_freq,
                              w_rtv=args.rtv_weight, highpass_ratio=args.highpass_ratio,
                              self_target=(args.pixel_target == "self")).to(device)
    print(f"[INFO] pixel_target={args.pixel_target}"
          + ("  (自重建恒等靶：预期几乎不去噪，仅作对照)" if args.pixel_target == "self" else ""))
    # 跨视图特征一致性：按 --feat_scales 选深度。model 返回 [enc1,enc2,enc3,bottleneck]=idx 0..3
    # 每项：(feat 索引, 通道数, 默认权重)；out3/bridge 为 encoder3/bottleneck 的别名
    _scale_info = {
        "encoder1": (0, FEAT_CHANNELS[0], 0.1),
        "encoder2": (1, FEAT_CHANNELS[1], 0.2),
        "encoder3": (2, FEAT_CHANNELS[2], 0.5), "out3": (2, FEAT_CHANNELS[2], 0.5),
        "bottleneck": (3, FEAT_CHANNELS[3], 1.0), "bridge": (3, FEAT_CHANNELS[3], 1.0),
    }
    feat_idx = [_scale_info[s][0] for s in args.feat_scales]          # 从 model 返回的 4 尺度里取哪些
    feat_ch = [_scale_info[s][1] for s in args.feat_scales]
    if args.feat_weights is not None:
        if len(args.feat_weights) != len(args.feat_scales):
            raise ValueError(f"--feat_weights({len(args.feat_weights)}) 长度须等于 --feat_scales({len(args.feat_scales)})")
        feat_w = list(args.feat_weights)
    else:
        feat_w = [_scale_info[s][2] for s in args.feat_scales]        # 默认权重
    print(f"[INFO] feature consistency scales={args.feat_scales} (idx={feat_idx}, ch={feat_ch}, w={feat_w})")
    criterion_feat = FeatureConsistencyLoss(
        channels=feat_ch, dim=args.feat_dim, pred_hidden=args.feat_pred_hidden,
        weights=feat_w, use_proj=bool(args.feat_use_proj),
        normalize_weights=bool(args.feat_normalize), pool=args.feat_pool).to(device)
    if args.feat_pool > 0:
        print(f"[INFO] 特征投影前池化到 {args.feat_pool}×{args.feat_pool}（区域级一致性，不约束高频）")
    if args.feat_normalize:
        print(f"[INFO] feat weights normalized to sum=1: {[round(w, 3) for w in criterion_feat.weights]}")
    print(f"[INFO] feat projection dims={criterion_feat.proj_dims} "
          f"(feat_dim={args.feat_dim}{' → 原生通道' if args.feat_dim <= 0 else ''}); "
          f"predictor hidden={criterion_feat.pred_hiddens}"
          f"{' (自动 dim//4)' if args.feat_pred_hidden is None else ' (手动指定)'}; "
          f"std 健康值≈1/√dim={[round(d**-0.5, 3) for d in criterion_feat.proj_dims]}")

    # 瓶颈辅助重建头（无 skip）。训练专用，参数与主网络一同优化，但不写进 model checkpoint。
    aux_dec = AuxDecoder(in_channels=FEAT_CHANNELS[3]).to(device) if args.w_aux > 0 else None
    if aux_dec is not None:
        print(f"[INFO] 启用瓶颈辅助重建 w_aux={args.w_aux}："
              f"L_aux = Charb(AuxDec(bridge(n1)), n2)，AuxDec 无 skip、推理时丢弃")

    aux_params = list(aux_dec.parameters()) if aux_dec is not None else []
    # predictor 是否独立、恒定 lr（SimSiam §4.2：h 应持续跟上最新表征，不应被强制收敛）
    pred_params = list(criterion_feat.preds.parameters())
    pred_ids = {id(p) for p in pred_params}
    if args.feat_pred_constant_lr > 0:
        main_params = [p for p in list(model.parameters()) + list(criterion_feat.parameters()) + aux_params
                       if id(p) not in pred_ids]
        opt_pred = optim.AdamW(pred_params, lr=args.feat_pred_constant_lr, weight_decay=1e-4)
        print(f"[INFO] predictor 使用独立优化器，恒定 lr={args.feat_pred_constant_lr}（不参与 OneCycle）")
    else:
        main_params = list(model.parameters()) + list(criterion_feat.parameters()) + aux_params
        opt_pred = None
    optimizer = optim.AdamW(main_params, lr=args.lr_max, weight_decay=1e-4)
    scheduler = build_onecycle(optimizer, len(train_loader), args)

    total_steps = args.epochs * len(train_loader)
    whiten_start = int(args.whiten_start_frac * total_steps)
    print(f"[INFO] total_steps={total_steps}, whitening starts at step {whiten_start} "
          f"(frac={args.whiten_start_frac}); train batches/epoch={len(train_loader)}")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); criterion_feat.train()   # 含投影/预测头的 BN
        running = {}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for n1, n2 in pbar:
            n1 = n1.to(device, non_blocking=True)
            n2 = n2.to(device, non_blocking=True)
            f_n1, feats1 = model(n1, return_feats=True)
            f_n2, feats2 = model(n2, return_feats=True)
            use_white = global_step >= whiten_start
            loss, logs = criterion(f_n1, n1, f_n2, n2, use_whitening=use_white)
            feats1_sel = [feats1[i] for i in feat_idx]              # 按 --feat_scales 选深度
            feats2_sel = [feats2[i] for i in feat_idx]
            feat_loss, feat_stds = criterion_feat(feats1_sel, feats2_sel)   # 跨视图特征一致性
            loss = loss + args.w_feat * feat_loss
            logs["feat"] = float(feat_loss.detach())

            if aux_dec is not None:                                # bridge 恒为 feats[3]，与 --feat_scales 无关
                aux_out = aux_dec(feats1[3], n1.shape[-2:])        # 只从瓶颈重建，无任何 skip
                aux_loss = criterion.charb(aux_out, n2)            # N2N 式靶子：独立噪声的兄弟帧
                loss = loss + args.w_aux * aux_loss
                logs["aux"] = float(aux_loss.detach())
            for si, s in enumerate(feat_stds):
                logs[f"std{si}"] = s

            optimizer.zero_grad(set_to_none=True)
            if opt_pred is not None:
                opt_pred.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion_feat.parameters()) + aux_params, args.grad_clip)
            optimizer.step()
            if opt_pred is not None:
                opt_pred.step()      # 恒定 lr，不走 scheduler
            scheduler.step()
            global_step += 1

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            pbar.set_postfix({"loss": f"{float(loss.detach()):.5f}", "feat": f"{logs['feat']:.4f}",
                              "std": "/".join(f"{s:.3f}" for s in feat_stds),
                              "lr": f"{scheduler.get_last_lr()[0]:.2g}"})

        n = max(1, len(train_loader))
        avg = {k: v / n for k, v in running.items()}
        save_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pth")
        state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(state, save_path)
        if args.save_feat_head:      # projector/predictor/aux 都不属于去噪模型，单独存
            torch.save(criterion_feat.state_dict(),
                       os.path.join(args.save_dir, f"feat_head_epoch_{epoch}.pth"))
            if aux_dec is not None:
                torch.save(aux_dec.state_dict(), os.path.join(args.save_dir, f"aux_dec_epoch_{epoch}.pth"))
        print(f"[EPOCH {epoch}] " + " ".join(f"{k}={avg[k]:.5f}" for k in
              ("total", "rec", "diff", "rtv", "feat", "aux", "std0", "std1", "std2", "std3", "white") if k in avg)
              + f"  saved={save_path}")


if __name__ == "__main__":
    train(parse_args())
