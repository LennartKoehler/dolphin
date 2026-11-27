# Configuration Parameters

The JSON config format is documented in the top-level [README](../README.md#json-configuration).
The files in this directory are working examples:

| File | Purpose |
|---|---|
| `default_config.json` | Standard deconvolution with a Gaussian PSF |
| `default_psf_config_gaussian.json` | Gaussian PSF config only |
| `default_psf_gibsonlanni.json` | Gibson-Lanni PSF config only |
| `gibson_lanni_config.json` | Deconvolution with a Gibson-Lanni widefield PSF |
| `gibson_lanni_confocal_config.json` | Confocal Gibson-Lanni PSF (finite pinhole) example |
| `labeled_image_config.json` | Labeled-image deconvolution (`labeled_image` + `label_psf_map`) |

Run a combined config with:

```bash
./dolphin deconvolution -c default_config.json
```

All parameters are also exposed as CLI flags; run `./dolphin deconvolution --help` for the
authoritative, auto-generated list.
