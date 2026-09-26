"""Dataset-sized YOLO batches and measured optimizer updates, independent of architecture."""
import hashlib
import math


def training_batch_policy(options, training_images, *, resume=False, checkpoint_args=None):
    """Keep full-size defaults; do not silently change a resumed optimizer schedule."""
    if type(training_images) is not int or training_images < 1:
        raise ValueError("Training needs at least one image")
    effective = dict(options)
    policy = {"name": "dataset_bounded_accumulation_v1", "training_images": training_images,
              "requested_batch": options["batch"], "resume_schedule_preserved": resume}
    if resume and checkpoint_args is not None:
        # Upstream permits a batch override on resume. Retain the checkpoint's
        # effective batch, which can be smaller than the original UI request.
        effective['batch'] = checkpoint_args.get('batch', options['batch'])
        effective['nbs'] = checkpoint_args.get('nbs', 64)
    elif not resume:
        effective["batch"] = min(options["batch"], training_images)
        effective["nbs"] = min(64, training_images)
    return effective, policy


def _weights_digest(parameters):
    import torch
    digest = hashlib.sha256()
    for name, parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        if not torch.isfinite(value).all().item():
            raise RuntimeError("Training produced nonfinite model weights")
        digest.update(repr((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TrainingUpdateAudit:
    """Observe optimizer calls, verify a numerical update, and hash run endpoints."""

    def __init__(self, optimizer, model):
        self.parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not self.parameters:
            raise RuntimeError("Training has no learnable parameters; check the model's frozen layers")
        self.initial_sha256 = _weights_digest(self.parameters)
        self.steps = 0
        self.positive_lr_steps = 0
        self._positive_lr = False
        self._before_parameters = []
        self.verified_parameter = None
        self.update_checks = 0
        self._parameter_names = {id(p): name for name, p in self.parameters}
        self._handles = [optimizer.register_step_pre_hook(self._before_step),
                         optimizer.register_step_post_hook(self._after_step)]

    def _before_step(self, optimizer, args, kwargs):
        rates = [float(group["lr"]) for group in optimizer.param_groups]
        if any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise RuntimeError("Training optimizer has an invalid learning rate")
        self._positive_lr = any(rate > 0 for rate in rates)
        # Stop taking snapshots after witnessing one real numerical optimizer update.
        # Initialization/finalization and signed-zero changes must not count as learning.
        self._before_parameters = []
        if self._positive_lr and self.verified_parameter is None:
            for group in optimizer.param_groups:
                if float(group['lr']) <= 0:
                    continue
                for parameter in group['params']:
                    if id(parameter) in self._parameter_names and parameter.grad is not None:
                        self._before_parameters.append((parameter, parameter.detach().cpu().clone()))

    def _after_step(self, optimizer, args, kwargs):
        import torch
        self.steps += 1
        self.positive_lr_steps += int(self._positive_lr)
        if self._before_parameters:
            self.update_checks += 1
            for parameter, before in self._before_parameters:
                if not torch.equal(before, parameter.detach().cpu()):
                    self.verified_parameter = self._parameter_names[id(parameter)]
                    break
        self._before_parameters = []

    def counts(self):
        return {"optimizer_steps": self.steps, "positive_lr_steps": self.positive_lr_steps}

    def finish(self):
        final_sha256 = _weights_digest(self.parameters)
        state_changed = final_sha256 != self.initial_sha256
        changed = self.verified_parameter is not None and state_changed
        return {**self.counts(), "learnable_weights_changed": changed,
                "optimizer_weight_change_verified": self.verified_parameter is not None,
                "verified_parameter": self.verified_parameter, "update_checks": self.update_checks,
                "trainable_state_changed": state_changed,
                "initial_trainable_sha256": self.initial_sha256,
                "final_trainable_sha256": final_sha256,
                "learnable_parameters": sum(p.numel() for _, p in self.parameters),
                "passed": self.positive_lr_steps > 0 and changed}

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._before_parameters = []


def require_training_updates(report):
    if (not report.get("passed") or report.get("positive_lr_steps", 0) < 1
            or not report.get("learnable_weights_changed") or not report.get("optimizer_weight_change_verified")):
        raise RuntimeError(
            "Training finished without an effective weight update. No new model was registered. "
            "Start a new training job with the current small-dataset settings; do not resume an "
            "old short run with an unsuitable schedule. Check the learning rate and frozen layers "
            "if the problem persists."
        )
