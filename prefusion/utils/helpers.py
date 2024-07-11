def prepare_model_cfg(model_cfg_base_path, model_cfg_override_dict):
    model_cfg = ModelConfiguration.from_dict({})
    if model_cfg_base_path is not None:
        model_cfg = ModelConfiguration.from_yaml(model_cfg_base_path)
    if model_cfg_override_dict:
        model_cfg_override = ModelConfiguration.from_dict(model_cfg_override_dict)
        model_cfg.update(model_cfg_override)
    return model_cfg


def prepare_dictionary(dictionary_base_path, dictionary_override_dict):
    dictionary = ExtensibleDictionary.from_dict({})
    if dictionary_base_path is not None:
        dictionary = ExtensibleDictionary.from_yaml(dictionary_base_path)
    if dictionary_override_dict:
        dictionary_override = ExtensibleDictionary.from_dict(dictionary_override_dict)
        dictionary.update(dictionary_override)
    return dictionary


def prepare_transforms(transform_cfg):
    train_transforms = []
    val_transforms = []
    eval_transforms = []

    if transform_cfg is None:
        default_transforms = [ToRGB(), GetData()]
        return {"TRAIN": default_transforms, "VALIDATION": default_transforms, "EVALUATION": default_transforms}

    for transform_cls_name, params in deepcopy(transform_cfg).items():
        transform_cls = getattr(T, transform_cls_name)
        assert (
            transform_cls in T.available_transforms
        ), f"{transform_cls.__name__} not in available transforms: {T.available_transforms}"
        try:
            phase = params.pop("phase", "all")
        except AttributeError:  # which means params is None
            params = {}
            phase = "all"
        transform = transform_cls(**params)
        if phase.lower() in ["train", "all", "train_val", "train_val_eval"]:
            train_transforms.append(transform)
        if phase.lower() in ["val", "validation", "validate", "all", "train_val", "train_val_eval", "val_eval"]:
            val_transforms.append(transform)
        if phase.lower() in ["eval", "evaluation", "evaluate", "all", "train_val_eval", "val_eval"]:
            eval_transforms.append(transform)

    train_transforms = [ToRGB()] + train_transforms + [GetData()]
    val_transforms = [ToRGB()] + val_transforms + [GetData()]
    eval_transforms = [ToRGB()] + eval_transforms + [GetData()]
    return {"TRAIN": train_transforms, "VALIDATION": val_transforms, "EVALUATION": eval_transforms}


def prepare_datasets(dataset_cfg, phase=None):
    supported_phases = ["TRAIN", "VALIDATION", "EVALUATION"] if phase is None else [phase]

    def _make_dataset(_d_cfg):
        dataset = {k: None for k in supported_phases}
        transform = {k: None for k in supported_phases}
        dictionary = prepare_dictionary(_d_cfg.DICTIONARY.BASE, _d_cfg.DICTIONARY.OVERRIDE)
        all_transforms = prepare_transforms(_d_cfg.TRANSFORMS)
        *dataset_str_parts, dataset_class_str = _d_cfg.CLASS.split(".")
        dataset_class = getattr(importlib.import_module(".".join(dataset_str_parts)), dataset_class_str)

        for ph in supported_phases:
            if _d_cfg.ARGS.get(ph):
                dataset[ph] = dataset_class(_d_cfg.NAME, dictionary, **(_d_cfg.ARGS[ph] or {}))
                # add transform to dataset here!!!
                # dataset[ph] = dataset_class(_d_cfg.NAME, dictionary, **(_d_cfg.ARGS[ph] or {}), transform=all_transforms[ph])
                transform[ph] = all_transforms[ph]
        return dataset, transform

    datasets, transforms = [], []
    for d_cfg in dataset_cfg:
        d, t = _make_dataset(d_cfg)
        datasets.append(d)
        transforms.append(t)

    return datasets, transforms


def prepare_datasets_within_transforms(dataset_cfg, batch_size, phase=None, drop_last=True):
    supported_phases = ["TRAIN", "VALIDATION", "EVALUATION"] if phase is None else [phase]

    def _make_dataset(_d_cfg, bs):
        dataset = {k: None for k in supported_phases}
        # dataset = {}
        dictionary = prepare_dictionary(_d_cfg.DICTIONARY.BASE, _d_cfg.DICTIONARY.OVERRIDE)
        all_transforms = prepare_transforms(_d_cfg.TRANSFORMS)
        *dataset_str_parts, dataset_class_str = _d_cfg.CLASS.split(".")
        dataset_class = getattr(importlib.import_module(".".join(dataset_str_parts)), dataset_class_str)
        for ph in supported_phases:
            if _d_cfg.ARGS.get(phase):
                dataset[ph] = dataset_class(
                    _d_cfg.NAME, dictionary, **(_d_cfg.ARGS[ph] or {}), 
                    transforms=all_transforms[ph], batch_size=bs, 
                    shuffle=ph in ["TRAIN"], drop_last=drop_last
                )
        return dataset

    datasets = []
    for d_cfg in dataset_cfg:
        # print(d_cfg)
        d = _make_dataset(d_cfg, batch_size)
        datasets.append(d)
        # for ph in supported_phases:
        #     if d[ph] is not None:
        #         datasets.append(d)

    return datasets