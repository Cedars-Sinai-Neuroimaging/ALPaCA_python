#!/usr/bin/env python3
"""
Run ALPaCA inference pipeline.

Two modes:

1. Explicit paths:
    alpaca-run \
        --t1 /path/to/t1.nii.gz \
        --flair /path/to/flair.nii.gz \
        --epi /path/to/epi_mag.nii.gz \
        --phase /path/to/epi_phase.nii.gz \
        --labels /path/to/lesion_labels.nii.gz \
        --output /path/to/output

2. Auto-detect:
    alpaca-run \
        --subject-dir /path/to/subject/session_date \
        --output /path/to/output
"""

import argparse
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='Can\'t initialize NVML')

from ..preprocessing.pipeline import run_alpaca


def find_file(directory, patterns):
    """Find first file matching any of the patterns."""
    directory = Path(directory)
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return str(matches[0])
    return None


def auto_detect_files(subject_dir):
    """Auto-detect MRI files from subject directory."""
    subject_dir = Path(subject_dir)

    patterns = {
        't1': ['*_T1_MTTE.nii.gz', 'T1_MTTE.nii.gz', 't1.nii.gz', 'T1.nii.gz', 't1_final.nii.gz'],
        'flair': ['*_FL_MTTE.nii.gz', 'FLAIR_MTTE.nii.gz', 'flair.nii.gz', 'FLAIR.nii.gz', 'flair_final.nii.gz'],
        'epi': ['*_T2star_mag_MTTE.nii.gz', '*_T2star_MTTE.nii.gz',
                'epi_mag.nii.gz', 'EPI_mag.nii.gz', 't2star_mag.nii.gz', 'epi_final.nii.gz'],
        'phase': ['*_T2star_phase_unwrapped_MTTE.nii.gz',
                  'epi_phase_unwrapped.nii.gz', 'epi_phase.nii.gz',
                  't2star_phase_unwrapped.nii.gz', 'phase.nii.gz', 'phase_final.nii.gz'],
        'labels': ['Lesion_Index_spectral.nii.nii.gz', 'Lesion_Index_Spectral.nii.nii.gz',
                   'Lesion_Index_spectral.nii.gz', 'lesion_labels.nii.gz',
                   'lesion_mask.nii.gz', 'labeled_candidates.nii.gz'],
        'eroded': ['eroded_candidates.nii.gz', 'eroded_labels.nii.gz']
    }

    files = {}
    for key, pattern_list in patterns.items():
        found = find_file(subject_dir, pattern_list)
        if found:
            files[key] = found

    return files


def main():
    parser = argparse.ArgumentParser(
        description='ALPaCA: Automated Lesion, PRL, and CVS Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect files
  alpaca-run --subject-dir /path/to/subject -o results/

  # Specify files explicitly
  alpaca-run --t1 t1.nii.gz --flair flair.nii.gz --epi epi.nii.gz \\
             --phase phase.nii.gz --labels labels.nii.gz -o results/
"""
    )

    # Input mode
    input_group = parser.add_argument_group('Input (choose one)')
    mode = input_group.add_mutually_exclusive_group(required=True)
    mode.add_argument('--subject-dir', metavar='DIR',
                      help='Auto-detect files from directory')
    mode.add_argument('--t1', metavar='FILE',
                      help='T1 image (requires --flair --epi --phase --labels)')

    # Explicit mode files
    explicit_group = parser.add_argument_group('Explicit mode (when using --t1)')
    explicit_group.add_argument('--flair', metavar='FILE', help='FLAIR image')
    explicit_group.add_argument('--epi', metavar='FILE', help='EPI magnitude')
    explicit_group.add_argument('--phase', metavar='FILE', help='EPI phase (unwrapped)')
    explicit_group.add_argument('--labels', metavar='FILE', help='Lesion labels')
    explicit_group.add_argument('--eroded-labels', metavar='FILE', help='Pre-eroded labels (optional)')

    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True, metavar='DIR',
                              help='Output directory')
    output_group.add_argument('--model-dir', metavar='DIR',
                              help='Model directory (default: <package>/models)')

    # Inference parameters
    inference_group = parser.add_argument_group('Inference parameters')
    inference_group.add_argument('--n-patches', type=int, default=20, metavar='N',
                                 help='Patches per lesion (default: 20)')
    inference_group.add_argument('--n-models', type=int, default=10, metavar='N',
                                 help='CV models to use (default: 10)')
    inference_group.add_argument('--no-rotate', action='store_true',
                                 help='Disable patch rotation')
    inference_group.add_argument('--seed', type=int, metavar='N',
                                 help='Random seed')
    inference_group.add_argument('--return-prob-maps', action='store_true',
                                 help='Save probability maps')

    # Thresholds
    threshold_group = parser.add_argument_group('Thresholds')
    threshold_group.add_argument('--lesion-threshold', default='youdens_j',
                                 choices=['youdens_j', 'specificity', 'sensitivity'],
                                 help='Lesion threshold (default: youdens_j)')
    threshold_group.add_argument('--prl-threshold', default='youdens_j',
                                 choices=['youdens_j', 'specificity', 'sensitivity'],
                                 help='PRL threshold (default: youdens_j)')
    threshold_group.add_argument('--cvs-threshold', default='youdens_j',
                                 choices=['youdens_j', 'specificity', 'sensitivity'],
                                 help='CVS threshold (default: youdens_j)')

    # Other
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress')

    args = parser.parse_args()

    # Determine mode and get file paths
    if args.subject_dir:
        if args.verbose:
            print(f"Auto-detecting files from: {args.subject_dir}")

        files = auto_detect_files(args.subject_dir)

        required = ['t1', 'flair', 'epi', 'phase', 'labels']
        missing = [k for k in required if k not in files]

        if missing:
            print(f"Error: Could not auto-detect: {missing}")
            print(f"Searched in: {args.subject_dir}")
            print("Use explicit mode: --t1, --flair, --epi, --phase, --labels")
            return 1

        if args.verbose:
            for key in required + ['eroded']:
                if key in files:
                    print(f"  {key:8s}: {Path(files[key]).name}")

        t1_path = files['t1']
        flair_path = files['flair']
        epi_path = files['epi']
        phase_path = files['phase']
        labels_path = files['labels']
        eroded_path = files.get('eroded')

    else:
        # Explicit mode
        required_args = ['t1', 'flair', 'epi', 'phase', 'labels']
        missing = [arg for arg in required_args if getattr(args, arg) is None]

        if missing:
            print(f"Error: Missing required arguments: --{', --'.join(missing)}")
            return 1

        t1_path = args.t1
        flair_path = args.flair
        epi_path = args.epi
        phase_path = args.phase
        labels_path = args.labels
        eroded_path = args.eroded_labels

    # Validate paths exist
    for name, path in [('t1', t1_path), ('flair', flair_path),
                       ('epi', epi_path), ('phase', phase_path),
                       ('labels', labels_path)]:
        if not Path(path).exists():
            print(f"Error: {name} file not found: {path}")
            return 1

    if eroded_path and not Path(eroded_path).exists():
        print(f"Error: eroded-labels file not found: {eroded_path}")
        return 1

    if args.model_dir and not Path(args.model_dir).exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        return 1

    # Run ALPaCA pipeline
    try:
        results = run_alpaca(
            t1=t1_path,
            flair=flair_path,
            epi=epi_path,
            phase=phase_path,
            labeled_candidates=labels_path,
            eroded_candidates=eroded_path,
            model_dir=args.model_dir,
            output_dir=args.output,
            lesion_priority=args.lesion_threshold,
            prl_priority=args.prl_threshold,
            cvs_priority=args.cvs_threshold,
            n_patches=args.n_patches,
            n_models=args.n_models,
            rotate_patches=not args.no_rotate,
            return_probabilities=args.return_prob_maps,
            random_seed=args.seed,
            verbose=args.verbose
        )

        if args.verbose:
            print(f"\n✓ Pipeline completed successfully")
            print(f"✓ Results saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
