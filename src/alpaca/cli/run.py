#!/usr/bin/env python3
"""
Run ALPaCA inference pipeline.

Two modes for lesion candidates:

1. Using pre-labeled candidates:
    alpaca-run \\
        --t1 /path/to/t1.nii.gz \\
        --flair /path/to/flair.nii.gz \\
        --epi /path/to/epi_mag.nii.gz \\
        --phase /path/to/epi_phase.nii.gz \\
        --labels /path/to/lesion_labels.nii.gz \\
        --output /path/to/output

2. Using a probability map (candidates will be generated):
    alpaca-run \\
        --t1 /path/to/t1.nii.gz \\
        --flair /path/to/flair.nii.gz \\
        --epi /path/to/epi_mag.nii.gz \\
        --phase /path/to/epi_phase.nii.gz \\
        --prob-map /path/to/prob_map.nii.gz \\
        --output /path/to/output
"""

import argparse
import sys
from pathlib import Path

from ..logger import log, set_log_level


def main():
    parser = argparse.ArgumentParser(
        description='ALPaCA: Automated Lesion, PRL, and CVS Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using pre-labeled lesion candidates
  alpaca-run --t1 t1.nii.gz --flair flair.nii.gz --epi epi.nii.gz \\
             --phase phase.nii.gz --labels labels.nii.gz -o results/

  # Using a probability map to generate lesion candidates
  alpaca-run --t1 t1.nii.gz --flair flair.nii.gz --epi epi.nii.gz \\
             --phase phase.nii.gz --prob-map prob_map.nii.gz -o results/
"""
    )

    # Input files
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('--t1', metavar='FILE', required=True, help='T1 image')
    input_group.add_argument('--flair', metavar='FILE', required=True, help='FLAIR image')
    input_group.add_argument('--epi', metavar='FILE', required=True, help='EPI magnitude')
    input_group.add_argument('--phase', metavar='FILE', required=True, help='EPI phase (unwrapped)')
    input_group.add_argument('--eroded-labels', metavar='FILE', help='Pre-eroded labels (optional)')

    # Candidate source
    candidate_group = parser.add_argument_group('Candidate source (choose one)')
    source = candidate_group.add_mutually_exclusive_group(required=True)
    source.add_argument('--labels', metavar='FILE', help='Lesion labels file')
    source.add_argument('--prob-map', metavar='FILE', help='Lesion probability map (will be labeled)')

    # Candidate generation
    gen_group = parser.add_argument_group('Candidate generation (when using --prob-map)')
    gen_group.add_argument('--candidate-threshold', type=float, default=0.05, metavar='FLOAT',
                           help='Threshold for lesion probability map (default: 0.05)')

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
    inference_group.add_argument('--batch-size', type=int, default=20, metavar='N',
                                 help='Patches per batch (default: 20)')
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
    log_group = parser.add_argument_group('Logging')
    verbosity = log_group.add_mutually_exclusive_group()
    verbosity.add_argument('-q', '--quiet', action='store_true', help="Show only critical errors")
    verbosity.add_argument('-v', '--verbose', action='store_true', help="Show all debug messages")
    parser.add_argument('--skip-normalization', action='store_true',
                        help='Skip image normalization step')

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        set_log_level("quiet")
    elif args.verbose:
        set_log_level("verbose")
    else:
        set_log_level("standard")

    # Custom validation for arguments
    if args.prob_map and args.eroded_labels:
        log.error("The --eroded-labels option can only be used with --labels, not with --prob-map.")
        log.error("When using --prob-map, lesion candidates and their eroded versions are generated automatically.")
        return 1

    # Validate paths exist
    paths_to_check = {
        't1': args.t1,
        'flair': args.flair,
        'epi': args.epi,
        'phase': args.phase,
        'labels': args.labels,
        'prob_map': args.prob_map,
        'eroded_labels': args.eroded_labels,
        'model_dir': args.model_dir
    }

    for name, path in paths_to_check.items():
        if path and not Path(path).exists():
            log.error(f"{name} file/directory not found: {path}")
            return 1


    # Run ALPaCA pipeline
    try:
        from ..processing import run_alpaca

        results = run_alpaca(
                t1=args.t1,
                flair=args.flair,
                epi=args.epi,
                phase=args.phase,
                labeled_candidates=args.labels,
                prob_map=args.prob_map,
                eroded_candidates=args.eroded_labels,
                skip_normalization=args.skip_normalization,
                candidate_threshold=args.candidate_threshold,
                model_dir=args.model_dir,
                output_dir=args.output,
                lesion_priority=args.lesion_threshold,
                prl_priority=args.prl_threshold,
                cvs_priority=args.cvs_threshold,
                n_patches=args.n_patches,
                n_models=args.n_models,
                rotate_patches=not args.no_rotate,
                return_probabilities=args.return_prob_maps,
                random_seed=args.seed
        )

        if results:
            log.info(f"[bold green]Pipeline completed successfully.[/bold green]")
        else:
            log.warning("Pipeline finished, but no lesions were processed.")

        return 0

    except Exception as e:
        log.error(f"An unexpected error occurred: {e}")
        log.debug(e, exc_info=True) # Full traceback on verbose
        return 1


if __name__ == "__main__":
    sys.exit(main())
