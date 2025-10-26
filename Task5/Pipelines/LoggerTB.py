# tb_logger.py  ────────────────────────────────────────────────────────────
from typing   import List, Optional, Dict
from pathlib  import Path
import os

import random, torch
import torchvision.transforms as T
from torchvision.utils import make_grid
from PIL import Image
import torchvision.transforms as transforms
from Tools.Utilities import cv2torch, TransformToDisplay
from Pipelines.DatasetWnet import transformTrainZero, transformTrainMinor, transformTrainHalf, transformTrainFull, transformSingleContentGT
from PIL import Image, ImageDraw, ImageFont
from Tools.Utilities import BaisedRandomK,MakeGifFromPngs
import math
from typing import Tuple

# -------------------------------------------------------------------------
# constants you already define elsewhere – just import them in Trainer
DISP_CONTENT_STYLE_NUM   = 5
NUM_FRAME_DISPLAY_IN_TB  = 5
NUM_SAMPLE_PER_EPOCH = 1000  # Number of samples processed per epoch
NUM_RECORD_NORMAL = 3
from Pipelines.Trainer import START_TRAIN_EPOCHS,INITIAL_TRAIN_EPOCHS
# RECORD_PCTG = NUM_SAMPLE_PER_EPOCH / NUM_RECORD_NORMAL  # Percentage of samples to record during training

# -------------------------------------------------------------------------

    

class TBLogger:
    
    def __init__(self,
                 writer,                       # SummaryWriter (already created in Trainer)
                 cfg,sessionLog):
        self.writer   = writer
        self.cfg      = cfg
        self.to_pil   = T.ToPILImage()
        self.sessionLog = sessionLog

    # ===============================================================
    # 仅做决策，不负责真正写 TensorBoard
    # ===============================================================
    @staticmethod
    def ShouldWriteSummaryTrain(
        *,
        idx: int,                   # 当前 batch 序号（从 0 开始）
        epoch: int,                 # 当前 epoch
        dataset_len: int,           # 该 epoch 总样本数
        batch_size: int,            # batch_size（不是 len(loader)）
        last_summary_prog: float,   # 上一次写 summary 时的 progress
        last_anim_prog: float,      # 上一次写 GIF 时的 progress
    ) -> Tuple[bool, bool, float]:
        
        RECORD_PCTG = float(NUM_SAMPLE_PER_EPOCH / NUM_RECORD_NORMAL)
        """
        逻辑判断器  
        返回 (need_summary, need_animation, progress)
        progress ∈ [0, NUM_SAMPLE_PER_EPOCH]
        """

        # ── 1. 进度计算（用 *浮点* n_batches） ────────────────────────────
        n_batches_float = dataset_len / batch_size             # 可能是小数
        last_batch_idx  = math.ceil(n_batches_float) - 1       # 真正的最后一个 idx
        progress        = idx * NUM_SAMPLE_PER_EPOCH / n_batches_float
        delta1          = progress - last_summary_prog
        delta2          = progress - last_anim_prog

        # ── 2. 各阶段 summary 触发阈值 ────────────────────────────────
        if epoch < START_TRAIN_EPOCHS:
            summary_thresh = RECORD_PCTG / 10                  # 最前期记录更密
        elif epoch < INITIAL_TRAIN_EPOCHS:
            summary_thresh = RECORD_PCTG / 5
        else:
            summary_thresh = RECORD_PCTG

        # ── 3. 是否该写 summary ──────────────────────────────────────
        need_summary = (
            idx in (0, last_batch_idx)          # 头/尾批必写
            or delta1 > summary_thresh          # 间隔到阈值也写
        )

        # ── 4. 是否该生成动画 / essay ─────────────────────────────────
        need_animation = (
            (epoch < START_TRAIN_EPOCHS and                     # 早期：头/尾/间隔 RECORD_PCTG
             (idx in (0, last_batch_idx) or delta2 > RECORD_PCTG))
            or
            (epoch >= START_TRAIN_EPOCHS and idx == last_batch_idx)  # 后期：只在尾批
        )

        return need_summary, need_animation, progress
    
    
    @staticmethod
    def ShouldWriteSummaryTest(
        *,
        idx: int,                   # 当前 batch 序号
        epoch: int,                 # 当前 epoch
        dataset_len: int,           # 该 epoch 样本总数
        batch_size: int,
        last_summary_prog: float,   # 上一次写 summary 的 progress
    ) -> tuple[bool, float]:
        RECORD_PCTG = float(NUM_SAMPLE_PER_EPOCH / NUM_RECORD_NORMAL)
        """
        返回 (need_summary, progress)

        - progress ∈ [0, NUM_SAMPLE_PER_EPOCH]
        - **不涉及** GIF / Animation 的判定（测试阶段原逻辑也没有）。
        """
        # ── 1. 进度计算（浮点 n_batches） ───────────────────────────────
        n_batches_float = dataset_len / batch_size
        last_batch_idx  = math.ceil(n_batches_float) - 1
        progress        = idx * NUM_SAMPLE_PER_EPOCH / n_batches_float
        delta           = progress - last_summary_prog

        # ── 2. summary 触发阈值 ───────────────────────────────────────
        if epoch < START_TRAIN_EPOCHS:
            thresh = RECORD_PCTG                     # 早期：RECORD_PCTG
        else:
            thresh = RECORD_PCTG * 2                 # 之后：RECORD_PCTG ×2

        # ── 3. 是否写 summary ──────────────────────────────────────
        need_summary = (
            idx in (0, last_batch_idx)     # 头/尾批
            or delta > thresh              # 超过阈值
        )

        return need_summary, progress
    
    def ClearGradRecords(self,grad):
        """Reset gradient records to 0 for all layers."""
        for idx1, (subName, subDict) in enumerate(grad.items()):
            for idx2, (key, value) in enumerate(grad[subName].items()):
                grad[subName][key].value = 0.0

                
                
                
    # =====================================================================
    #  main call
    # =====================================================================
    def Write2TB(self,
                    *,
                    generator, 
                    eval_contents,
                    eval_styles,
                    eval_gts,
                    eval_fakes,
                    step       : int,
                    mark       : str,
                    loss_dict,
                    grad_g:     Optional[float]  = None,
                    lr_g       : Optional[float]  = None,
                    grad_thresh: Optional[float]  = None,
                    essay_img  = [],                       # CHW tensor
                    gif_frames : Optional[List[Image.Image]] = None,
                    gif_tag    : str = "gif_display"):
        # ──────────────────────────────────────────────────────────
        # 1. big comparison grid
        grid = self.GridBuilder(eval_contents, eval_styles, eval_gts, eval_fakes)
        if mark == 'Train':
            self.writer.add_image("Images-Train",   grid, dataformats='CHW', global_step=step)
        elif 'Verifying@' in mark:
            self.writer.add_image(f"Images-{mark}", grid, dataformats='CHW', global_step=step)

        # extra essay preview
        if len(essay_img)>0:
            self.writer.add_image(f"Essay-Train", transforms.ToTensor()(essay_img[0]), dataformats='CHW', global_step=step)
            self.writer.add_image(f"Essay-Test", transforms.ToTensor()(essay_img[1]), dataformats='CHW', global_step=step)

        # optional gif (keeps only last NUM_FRAME_DISPLAY_IN_TB frames)
        if gif_frames:
            self.GifBuider(gif_frames, gif_tag, step)

        w = self.writer  # alias
        # ──────────────────────────────────────────────────────────
        # 2. reconstruction losses
        w.add_scalar(f"02-LossReconstruction-{mark}/L1",                        loss_dict['lossL1'],                step)
        w.add_scalar(f"02-LossReconstruction-{mark}/DeepPerceptualContentSum",  loss_dict['deepPerceptualContent'], step)
        w.add_scalar(f"02-LossReconstruction-{mark}/DeepPerceptualStyleSum",    loss_dict['deepPerceptualStyle'],   step)

        # ──────────────────────────────────────────────────────────
        # 3. generator losses
        w.add_scalar(f"05-LossGenerator-{mark}/SumLossG",               loss_dict['sumLossG'],                  step)
        w.add_scalar(f"05-LossGenerator-{mark}/ConstContentReal",       loss_dict['lossConstContentReal'],      step)
        w.add_scalar(f"05-LossGenerator-{mark}/ConstStyleReal",         loss_dict['lossConstStyleReal'],        step)
        w.add_scalar(f"05-LossGenerator-{mark}/ConstContentFake",       loss_dict['lossConstContentFake'],      step)
        w.add_scalar(f"05-LossGenerator-{mark}/ConstStyleFake",         loss_dict['lossConstStyleFake'],        step)
        w.add_scalar(f"05-LossGenerator-{mark}/CategoryRealContent",    loss_dict['lossCategoryContentReal'],   step)
        w.add_scalar(f"05-LossGenerator-{mark}/CategoryFakeContent",    loss_dict['lossCategoryContentFake'],   step)
        w.add_scalar(f"05-LossGenerator-{mark}/CategoryRealStyle",      loss_dict['lossCategoryStyleReal'],     step)
        w.add_scalar(f"05-LossGenerator-{mark}/CategoryFakeStyle",      loss_dict['lossCategoryStyleFake'],     step)


        # ──────────────────────────────────────────────────────────
        # 4. deep perceptual, per extractor
        if 'extractorContent' in self.cfg:
            for n,val in zip([e.name for e in self.cfg.extractorContent],
                             loss_dict['deepPerceptualContentList']):
                w.add_scalar(f"03-LossDeepPerceptual-ContentMSE-{mark}/{n}", val, step)

        if 'extractorStyle' in self.cfg:
            for n,val in zip([e.name for e in self.cfg.extractorStyle],
                             loss_dict['deepPerceptualStyleList']):
                w.add_scalar(f"04-LossDeepPerceptual-StyleMSE-{mark}/{n}", val, step)

        # ──────────────────────────────────────────────────────────
        # 5. grad + LR   (train only)
        if mark == 'Train':
            if grad_thresh is not None:
                w.add_scalar("01-GradientCheck-G/00-MinGradThreshold", grad_thresh, step)
            if lr_g is not None: w.add_scalar("00-LR/LR-G", lr_g, step)

            # per-layer gradient norms (values already accumulated by Trainer)
            self.ClearGradRecords(grad=grad_g)
            # >>> 先把「本 batch 的梯度范数」累加到 grad_g / grad_d
            for name, p in generator.named_parameters():
                if p.grad is None:               # 有的层可能没有梯度
                    continue
                parts = name.split('.')                 # ← 改这里
                if len(parts) < 2:                      # 极端情况保护
                    continue
                sub, layer = parts[0], parts[1]         # 与初始化保持一致
                grad_g[sub][layer].value += torch.norm(p.grad)
            for sub, layers in grad_g.items():
                for lyr, obj in layers.items():
                    tag = f"01-GradientCheck-G/{sub}-{lyr}"
                    w.add_scalar(tag, obj.value / max(obj.count, 1e-9) * lr_g, step)


        # ──────────────────────────────────────────────────────────
        self.writer.flush()

    # =====================================================================
    # helpers
    # =====================================================================
    def GridBuilder(self, contents, styles, gts, fakes):
        disp_c = min(DISP_CONTENT_STYLE_NUM, contents.shape[1])
        disp_s = min(DISP_CONTENT_STYLE_NUM, styles.shape[1])
        sel_c  = random.sample(range(contents.shape[1]), disp_c)
        sel_s  = random.sample(range(styles.shape[1]),   disp_s)

        rows = []
        for b in range(contents.shape[0]):
            diff  = torch.abs(fakes[b] - gts[b])
            parts = list(contents[b][sel_c].unsqueeze(1)) \
                  + [gts[b], diff, fakes[b]] \
                  + list(styles[b][sel_s].unsqueeze(1))
            rows.append(make_grid(parts,
                                  nrow = disp_c + disp_s + 3,
                                  normalize=True, scale_each=True))
        return make_grid(rows, nrow=1, normalize=True, scale_each=True)


    def GifBuider(self, frames: List[Image.Image], tag: str, step: int):
        frames = frames[-NUM_FRAME_DISPLAY_IN_TB:]
        duration = max(frames[0].info.get("duration",100), 10)
        fps      = max(1, min(5, round(1000./duration)))

        tensor_frames = [T.ToTensor()(f) for f in frames]
        vid = torch.stack(tensor_frames).unsqueeze(0)   # (1,T,C,H,W)
        self.writer.add_video(tag, vid, global_step=step, fps=fps)
        
        
    def WritingEssay(self, this_set, epoch_float, mark, generator):   
        generator.eval()
        with torch.no_grad():
            batch_size = self.cfg.trainParams.batchSize
            K = self.cfg.datasetConfig.availableStyleNum

            full_eval = this_set.evalContents
            full_size = (full_eval.shape[0] // batch_size) * batch_size
            trimmed = full_eval[:full_size].float().cuda()
            content_batches = list(torch.split(trimmed, batch_size, dim=0))

            remainder = full_eval[full_size:]
            if remainder.shape[0] > 0:
                content_batches.append(remainder.float().cuda())

            for idx1, style_label in enumerate(this_set.evalStyleLabels):
                style_data = this_set.styleFileDict[style_label]
                style_paths = [p for group in style_data for p in group]

                n = BaisedRandomK(K, len(style_paths))
                sampled = random.sample(style_paths, n)
                final_styles = sampled + random.choices(sampled, k=K - n) if n < K else sampled

                style_cat = torch.cat([
                    (cv2torch(p, transform=transformSingleContentGT) - 0.5) * 2 for p in final_styles
                ], dim=0).float().cuda()

                all_generated = []
                for content in content_batches:
                    style_expand = style_cat.unsqueeze(0).repeat(content.shape[0], 1, 1, 1)
                    blank = torch.zeros(content.shape[0], 1, 64, 64).float().cuda()
                    gen = generator(content, style_expand.reshape(-1, 1, 64, 64), blank, is_train=False)[4]
                    all_generated.append(gen)

                all_generated = torch.cat(all_generated, dim=0)

                fake_disp = TransformToDisplay(all_generated)
                real_disp = TransformToDisplay(this_set.evalGts[idx1][0])
                sep = Image.new("RGB", (3, fake_disp.height), (0, 0, 0))

                combined = Image.new("RGB", (fake_disp.width + 3 + real_disp.width, fake_disp.height))
                combined.paste(fake_disp, (0, 0))
                combined.paste(sep, (fake_disp.width, 0))
                combined.paste(real_disp, (fake_disp.width + 3, 0))

                draw = ImageDraw.Draw(combined)
                font = ImageFont.load_default()
                text = f"Font {style_label} Generated & GT @ Epoch {epoch_float:06.2f}"
                bbox = draw.textbbox((0, 0), text, font=font)
                x = combined.width - (bbox[2] - bbox[0]) - 10
                y = combined.height - (bbox[3] - bbox[1]) - 10
                draw.text((x, y), text, font=font, fill=(0, 0, 0))

                save_path = os.path.join(self.cfg.userInterface.trainImageDir, style_label.zfill(5))
                os.makedirs(save_path, exist_ok=True)
                combined.save(os.path.join(save_path, f"{mark}-Epoch-{epoch_float:06.2f}.png"))

            # Create GIFs from saved images per style
            for style in this_set.evalStyleLabels:
                styleDir = Path(os.path.join(self.cfg.userInterface.trainImageDir, style.zfill(5)))
                files = sorted((p for p in styleDir.rglob("*.png") if mark in p.name), 
                                key=lambda p: p.stat().st_mtime)
                MakeGifFromPngs(self.sessionLog, [str(p) for p in files],
                                os.path.join(self.cfg.userInterface.trainImageDir, f"Style-{style.zfill(5)}-{mark}.gif"))
            
            return combined