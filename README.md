# ALPaCA Python (Automated Segmentation of MS Lesions, PRLS, CVS)


Python implementation of the [ALPaCA](https://github.com/hufengling/ALPaCA) R package for MS lesion segmentation (Hu et al., 2025, *Imaging Neuroscience*). 

## Installation

```bash
cd ALPacA_python
pip install -e .
alpaca-download-models 
```

## Usage

**Input Requirements**

- T1, FLAIR, EPI mag, EPI phase
- Labled lesion candidates
- N4 bias corrected
- Co-registered
- Skull-stripped

**Outputs**

  - alpaca_mask.nii.gz - Segmentation mask
  - predictions.csv - Binary predictions (lesion/PRL/CVS) per lesion
  - probabilities.csv - Probability scores per lesion
  - uncertainties.csv - Uncertainty estimates per lesion
  - model_disagreement.csv - Disagreement between models per lesion

---
**Command Line**

```bash
# Auto-detect files
alpaca-run --subject-dir /path/to/subject --output results/

# Specify files
alpaca-run \
    --t1 t1.nii.gz \
    --flair flair.nii.gz \
    --epi epi_mag.nii.gz \
    --phase epi_phase.nii.gz \
    --labels lesion_labels.nii.gz \
    --output results/
```

**Python API**

```python
results = run_alpaca(
    t1='t1.nii.gz',
    flair='flair.nii.gz',
    epi='epi.nii.gz',
    phase='phase.nii.gz',
    labeled_candidates='labels.nii.gz',
    model_dir='models/',
    output_dir='results/'
)
```