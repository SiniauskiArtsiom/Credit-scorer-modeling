import json, yaml
with open("artifacts/selected_features.json") as f:
    sel = json.load(f)["features"]

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["selected_features"] = sel
with open("configs/config.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
