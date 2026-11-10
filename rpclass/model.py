# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""RPCLASS: the one object to create, train, inspect, predict with and export a classifier.

    from rpclass import RPCLASS
    model = RPCLASS('s')                                     # new model of a size: 's', 'm' (default), 'l', 'lt'
    model = RPCLASS('runs/train/exp/weights/best.pt')        # trained model: everything comes from the checkpoint
    model.train('parts')                                     # a folder: train/, val/ and test/, one folder per class
    results = model.predict(image, color='BGR')
    model.export()
"""
from pathlib import Path
from time import time as _clock

import torch
from torch.utils.flop_counter import FlopCounterMode

from .cfg.schema import DEFAULT_SIZE, SIZES, default_imgsz, derive, describe, input_size, model_config, size_of
from .data import ClassificationDataset, check_dataset, class_map, default_augment, upright_augment
from .data.check import SPLITS
from .engine.exporter import export_onnx
from .engine.predictor import predict as _predict
from .engine.trainer import Budget, _run_dir, train as _train
from .engine.validator import validate
from .nn import Classifier
from .utils.checkpoint import build_from_checkpoint, load_checkpoint


def _default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class RPCLASS:
    def __init__(self, model=DEFAULT_SIZE):
        """model: a size, 's', 'm' or 'l' (CNNs) or 'lt' (a transformer), a new model; or the path of a trained
        checkpoint (.pt)."""
        self.ckpt_path = None
        self.net = None            # built when training starts, or loaded from a checkpoint
        self.names = None
        self.imgsz = None
        self.data = None           # dataset the model was trained on
        self.keep_orientation = False   # trained without flips or turns (train's keep_orientation)
        if str(model).endswith('.pt'):
            ckpt = load_checkpoint(model)
            self.ckpt_path = Path(model)
            self.cfg = ckpt['config']
            self.names, self.imgsz, self.data = ckpt['names'], ckpt['imgsz'], ckpt.get('data')
            self.keep_orientation = bool(ckpt.get('keep_orientation', False))
            self.net = build_from_checkpoint(ckpt).eval()
        elif model in SIZES:
            self.cfg = model_config(model)
        else:
            raise ValueError(f"RPCLASS({model!r}): give a size ({', '.join(map(repr, SIZES))}) or a checkpoint (.pt)")

    def __repr__(self):
        state = f"trained, {len(self.names)} classes, {self.imgsz[0]}x{self.imgsz[1]}" if self.names else 'not trained'
        return f'RPCLASS({describe(self.cfg)}; {state})'

    # -------------------------------------------------------------------------------------------- inspect
    def info(self, imgsz=None, verbose=True):
        """Parameters (backbone and head) and GFLOPs at the given input size, else the trained one, else the size's
        default."""
        imgsz = input_size(imgsz or self.imgsz or default_imgsz(self.cfg))
        num_classes = len(self.names) if self.names else 2
        net = self.net if self.net is not None else Classifier(derive(self.cfg, num_classes, imgsz), pretrained=False)
        net = net.eval()
        head = sum(p.numel() for p in net.head_parameters())
        total = sum(p.numel() for p in net.parameters())
        device = next(net.parameters()).device
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            net(torch.zeros(1, 3, *imgsz, device=device))
        summary = {'backbone': total - head, 'head': head, 'total': total, 'gflops': fc.get_total_flops() / 1e9,
                   'imgsz': imgsz}
        if verbose:
            print(repr(self))
            print(f'input {imgsz[0]}x{imgsz[1]} · {num_classes} classes')
            for name in ('backbone', 'head'):
                print(f'  {name:16}{summary[name] / 1e6:8.2f} M')
            print(f"  {'total':16}{total / 1e6:8.2f} M   {summary['gflops']:.2f} GFLOPs")
        return summary

    # -------------------------------------------------------------------------------------------- train / validate
    def train(self, data, imgsz=None, epochs=100, patience=20, batch=32, device=None, augment=default_augment,
              name='exp', keep_orientation=None, pretrained=True, time=None, reserve=None, final_eval=True,
              project='runs/train', workers=None):
        """Train on a dataset folder (train/, val/, optional test/, one subfolder per class). Afterwards this object is
        the best model (runs/train/<name>/weights/best.pt).

        imgsz: int or [height, width], rounded up to multiples of 32; the exported model serves at this size too (the
            image is letterboxed into it, never deformed). None: the size this model was trained at (a checkpoint, or
            this object after a train); else the size's default, 384 for s, m and l, 448 for lt (docs/EXPERIMENTS.md).
        epochs / patience: after `patience` epochs without a higher macro-F1 on val the final learning-rate decay
            starts early, and training ends when it is done. None: never. A function patience(planned_epochs) -> int is
            evaluated again whenever the end of training moves (with `time`).
        batch: effective batch; if it does not fit in the GPU it is split into accumulation steps (shown in the log).
        augment: function(sample, ctx) -> sample applied to training images before the letterbox; by default
            rpclass.data.augment's: photometric distortion and the 8 orientations (flips, turns by 90 degrees). None:
            no augmentation. Any other function replaces it (contract in rpclass.data.dataset).
        keep_orientation: True when the orientation is part of the class (a part mounted upside down, a mirrored
            label): the default augmentation then never flips or turns an image (rpclass.data.augment.upright_augment).
            None: what this model was trained with (a checkpoint keeps it), else False. Saved in args.yaml and the
            checkpoint. A function of your own given as augment decides its own geometry: with it, True is an error.
        pretrained: a new model starts from the pretrained weights of its backbone (downloaded once). False: random
            weights. A model loaded from a checkpoint starts from it.
        time: seconds for the whole call, setup and the final evaluation included (estimated from the time each
            validation takes): training then ends when the time is up rather than after `epochs`, which becomes the
            most it may train. The final decay and the clean last epochs become shares of the time (the last 20 % and
            4 %), decided epoch by epoch from what each kind of epoch costs on this run; the warmup is sized to 5 % of
            the steps that fit. It stops between epochs, never inside one, and the last epoch is always a clean one
            from the best weights.
        reserve: seconds of `time` the caller needs after train returns (its own evaluation, export...), or a function
            reserve(stats) -> seconds evaluated after every epoch, with stats {'epoch_s', 'val_s', 'n_val',
            'val_s_per_image'} measured on this run.
        final_eval: False skips the final evaluation of the best weights on val and test: they are still loaded and
            adopted, 'val' and 'test' come back None.
        project / workers: the run goes to <project>/<name> (with a number added if it exists); dataloader workers
            (default: RPCLASS_WORKERS, else up to 8).
        Returns {'dir', 'best_epoch', 'val', 'test', 'planned_epochs', 'stop_reason', 'budget_overrun_s'}: the final
        metrics, the epochs the run ended at, why it ended ('EPOCHS_COMPLETED', 'TIME_BUDGET_REACHED' or
        'EARLY_STOPPED') and, with `time`, how many seconds it went past its share of the budget (negative: to spare).
        """
        explicit = keep_orientation is not None
        keep_orientation = bool(keep_orientation) if explicit else self.keep_orientation
        if keep_orientation:
            if augment in (default_augment, upright_augment):
                augment = upright_augment
            elif augment is not None and explicit:
                raise ValueError('train: keep_orientation=True picks the default augmentation without flips or turns; '
                                 'an augment function of your own decides its geometry itself, so give one or the '
                                 'other')
        if imgsz is None:
            imgsz = self.imgsz if self.imgsz is not None else default_imgsz(self.cfg)
        budget = Budget(time, _clock(), reserve) if time is not None else None
        if patience is not None and not callable(patience):
            patience = int(patience)
        return _train(self, data, input_size(imgsz), int(epochs), patience, int(batch), device or _default_device(),
                      augment, name, project=project, workers=workers, budget=budget, final_eval=final_eval,
                      pretrained=pretrained, keep_orientation=keep_orientation)

    def val(self, data=None, split='val', device=None, batch=32, name='exp'):
        """Metrics and plots (runs/val/<name>) on a split of the training dataset or of another dataset with the same
        classes (any order)."""
        self._require_trained('validate')
        data = data or self.data
        if data is None:
            raise ValueError('no dataset: pass data="<folder>"')
        ds = check_dataset(data)
        split = next((k for k, aliases in SPLITS.items() if split in aliases), split)     # 'validation' -> 'val'
        if not ds['records'].get(split):
            raise ValueError(f"the dataset has no images in a '{split}' split (train, val or test)")
        dataset = ClassificationDataset(ds['records'][split], self.imgsz, id_map=class_map(ds['names'], self.names))
        out = _run_dir('runs/val', name)
        metrics = validate(self.net.to(device or _default_device()), dataset, self.names, device or _default_device(),
                           batch, plots_dir=out)
        print(f'{split} of {data} · plots in {out}')
        if metrics.nonfinite:
            print(f'warning: {metrics.nonfinite} images gave non-finite scores; they count as wrong')
        print(metrics.table(self.names))
        return metrics

    def _adopt(self, net, names, imgsz, ckpt_path, data, keep_orientation=False):
        self.net, self.names, self.imgsz = net, names, list(imgsz)
        self.ckpt_path, self.data, self.keep_orientation = Path(ckpt_path), data, bool(keep_orientation)

    # -------------------------------------------------------------------------------------------- use
    def predict(self, source, color='RGB', device=None):
        """Class probabilities for an image array (with its color tag: RGB, BGR or GRAY), a file, a folder or a list.

        Returns one Result per image: .top1 (class id), .name, .conf, .probs (all classes), .top(k)."""
        self._require_trained('predict')
        return _predict(self.net, self.names, self.imgsz, source, color, device or _default_device())

    def export(self, output_dir=None):
        """model.onnx + config.json for serving (contract in rpclass/engine/exporter.py). Returns the folder."""
        self._require_trained('export')
        if output_dir is None:
            output_dir = (self.ckpt_path.with_name(f'{self.ckpt_path.stem}_export') if self.ckpt_path
                          else 'rpclass_export')
        return export_onnx(self.net, size_of(self.cfg) or self.cfg.backbone, self.names, self.imgsz, output_dir)

    def _require_trained(self, what):
        if self.net is None or not self.names:
            raise RuntimeError(f'cannot {what}: this model is not trained yet. Train it, or load a checkpoint (.pt).')
