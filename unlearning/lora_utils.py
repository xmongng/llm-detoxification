"""Resolve the parameter-update mode independently of the unlearning loss."""


def resolve_lora(method, enabled, rank):
    if enabled is not None and type(enabled) is not bool:
        raise ValueError('LoRA.enabled must be true, false, or null (method default).')
    if type(rank) is not int or rank < 0:
        raise ValueError('LoRA.r must be a nonnegative integer.')
    if enabled is True and rank == 0:
        raise ValueError('LoRA.enabled=true requires LoRA.r > 0.')
    if enabled is None:
        enabled = method not in ('grad_diff', 'simnpo', 'simnpo_gdr', 'codeeraser', 'prod')
    return bool(enabled and rank > 0)


def configure_trainable_model(model, enabled, lora_config=None, wrap_model=None):
    if enabled:
        if lora_config is None or wrap_model is None:
            raise ValueError('LoRA configuration and model wrapper are required.')
        return wrap_model(model, lora_config)
    # Adapter merging can leave frozen parameters behind. Full mode trains all.
    model.requires_grad_(True)
    return model


def parameter_summary(model, enabled):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(mode='lora' if enabled else 'full', total_parameters=total,
                trainable_parameters=trainable,
                trainable_percent=100 * trainable / total if total else 0.)
