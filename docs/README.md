# ESFM Project Page

This directory contains the GitHub Pages site for the ESFM paper, based on the [Nerfies](https://nerfies.github.io/) template.

## Enabling GitHub Pages

1. Push this `docs/` directory to the `main` branch of `swiss-ai/ESFM`.
2. On GitHub: **Settings → Pages → Source** → select **Deploy from a branch** → branch **`main`**, folder **`/docs`** → Save.
3. The site will be available at `https://swiss-ai.github.io/ESFM/` within a minute or two.

## File structure

```
docs/
├── index.html                # Main page
├── static/
│   ├── css/index.css         # Custom styles on top of Bulma
│   ├── js/index.js           # Carousel initialization
│   ├── images/               # Static figures (PNG/JPG) — see below
│   ├── videos/               # Animated content (MP4/GIF) — see below
│   └── pdfs/
│       └── ESFM_preprint.pdf # Drop the preprint PDF here
└── README.md                 # This file
```

All CSS/JS dependencies (Bulma, bulma-carousel, FontAwesome, Academic Icons) are loaded from CDNs, so there is nothing else to install.

## Media to provide

Drop your files at the paths below (filenames in `index.html` reference these exact names — rename either the files or the references in `index.html`).

### Required figures — `static/images/`

| Filename                                | Source in paper            | Description                                         |
| --------------------------------------- | -------------------------- | --------------------------------------------------- |
| `teaser_unified_framework.png`          | Figure 1                   | The hero figure showing CMIP6/ERA5/MODIS/Station → ESFM → forecasts. |
| `architecture_encoder.png`              | Figure 2                   | Encoder schematic with axial attention + Perceivers. |
| `architecture_nan_tokens.png`           | Figure 3                   | Partial vs. completely missing input → NaN tokens.   |
| `architecture_decoder.png`              | Figure 5                   | Decoder with AdaLN-Zero ensemble heads.              |
| `doksuri_intensity_track.png`           | Figure 7                   | Max wind velocity + track for Doksuri.               |
| `ssw_wind_velocity.png`                 | Figure 8                   | 10 hPa wind during the three 2024 SSW events.        |
| `ssw_stratosphere_troposphere.png`      | Figure 9                   | 500 hPa geopotential anomaly after Jan SSW.          |
| `multidecadal_stability.png`            | Figure 10                  | 25-year rollout of 2 m temperature over Europe.      |
| `physical_consistency_TQ.png`           | Figure 14                  | T–Q joint distribution with/without level masking.   |

### Animated content — `static/videos/`

The carousel uses `<video>` elements with `autoplay muted loop` (which lets them animate without user interaction). MP4 works best across browsers; GIFs are fine for the standalone figures.

| Filename                          | What you mentioned       | Suggested content                                          |
| --------------------------------- | ------------------------ | ---------------------------------------------------------- |
| `masked_era5_prediction.mp4`      | Masked ERA5              | Rollout of an ERA5 forecast with a continent masked at t=0. |
| `modis_pwv_prediction.mp4`        | MODIS satellite          | Sparse swath input → globally dense PWV forecast.           |
| `station_prediction.mp4`          | Station data             | Station observations → forecast at held-out stations.       |
| `doksuri_evolution.gif`           | (suggested addition)     | Animated version of the 4-row Figure 29 (ERA5 vs models).   |

If you only have GIFs, change the `<video>` tags in `index.html` to `<img src="...gif">`. Example:

```html
<!-- Replace this: -->
<video poster="" id="masked-era5" autoplay controls muted loop playsinline>
  <source src="./static/videos/masked_era5_prediction.mp4" type="video/mp4">
</video>

<!-- With this: -->
<img src="./static/videos/masked_era5_prediction.gif" alt="Masked ERA5 prediction">
```

### Preprint PDF

Drop the preprint at `static/pdfs/ESFM_preprint.pdf`. The "Paper" button in the hero links there.

## Updating the page

- **Authors / affiliations:** edit the hero section in `index.html`.
- **arXiv link:** replace `https://arxiv.org/abs/submit/7499478` with the public arXiv URL once it resolves.
- **Weights link:** the HuggingFace placeholder points to `https://huggingface.co/swiss-ai` — update once weights are uploaded.
- **BibTeX:** update once arXiv assigns the canonical ID.

## Optional additions

Sections that are easy to add and were suggested but not included by default:

- **Interactive before/after slider** for masked ERA5 (using e.g. [img-comparison-slider](https://img-comparison-slider.sneas.io/)).
- **Ensemble spread visualization** — an animation cycling through the 8 ensemble members for a single event.
- **"Try a forecast" widget** — a small JS form that lets users pick a date/region and links them to a Colab notebook.

If you want any of these added, ping in an issue on the repo.

## Credits

Page template adapted from [Nerfies](https://github.com/nerfies/nerfies.github.io) (CC BY-SA 4.0).
