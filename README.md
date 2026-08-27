# ZHMolRPep

ZHMolRPep is a single-sequence deep-learning framework for RNA–peptide complex structure prediction. It combines nucleotide-level RiNALMo representations, residue-level ESM-2 representations, monomer-derived persistent Stanley–Reisner theory (PSRT) descriptors, a two-dimensional residual convolutional contact-prediction network, and restraint-guided PyRosetta assembly.

The repository contains the model code, homology-controlled dataset splits, pretrained model weights, and predicted three-dimensional complex structures associated with the ZHMolRPep study.

## Repository structure

```text
ZHMolRPep/
├── README.md
├── code/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── build_3d_complex.py
│   ├── predict_rna.py
│   ├── predict_peptide.py
│   ├── extract_rinalmo_features.py
│   ├── extract_esm2_features.py
│   ├── extract_3d_psrt_monomer.py
│   ├── psrt.py
│   └── best_zhmoltoporpi_model.pth
├── dataset/
│   ├── train.zip
│   ├── val.zip
│   └── test.zip
└── result/
    └── mypredict_3D.zip
```

The archived dataset is divided into 709 training, 39 validation, and 59 held-out test RNA–peptide complexes. `mypredict_3D.zip` contains the predicted three-dimensional complex models.

> **Filename note:** if the downloaded files are named `psrt(1).py` and `extract_3d_psrt_monomer(1).py`, rename them to `psrt.py` and `extract_3d_psrt_monomer.py`, respectively. The PSRT extraction script imports `PH` from `psrt.py`.

## Method overview

1. RNA and peptide monomer structures are predicted independently using RhoFold and ESMFold, respectively.
2. RiNALMo-Giga and ESM-2 3B generate per-nucleotide and per-residue contextual representations.
3. PSRT descriptors are calculated independently from the predicted RNA and peptide monomer structures over 50 filtration values from 0 to 15 Å.
4. A two-dimensional residual convolutional network predicts nucleotide–residue contact probabilities.
5. High-confidence contacts are converted into atom-pair distance restraints for PyRosetta FastRelax assembly.

## Requirements

The code was written in Python and requires a CUDA-capable GPU for practical extraction of the ESM-2 3B and RiNALMo-Giga features. Install a PyTorch build compatible with the local CUDA installation, followed by the remaining Python dependencies:

```bash
pip install numpy pandas tqdm scikit-learn requests biopython gudhi transformers multimolecule
```

The following external software and model checkpoints must be installed separately:

- PyTorch with CUDA support
- ESM-2 3B (`esm2_t36_3B_UR50D`)
- RiNALMo-Giga
- RhoFold and its pretrained checkpoint
- PyRosetta

PyRosetta is distributed separately and is not installed by the command above. The peptide-monomer script calls the public ESMFold API and therefore requires internet access.

## Input conventions

### Paired FASTA files

The language-model feature scripts expect one paired FASTA file per RNA–peptide complex. The two headers must end in `_RNA` and `_Peptide`:

```text
>example_RNA
AUGCGU...
>example_Peptide
MKR...
```

The FASTA filename, without `.fasta`, is used as the pair identifier.

### CSV input for monomer prediction

`predict_rna.py` and `predict_peptide.py` read `dataset_split_culled.csv`. The scripts accept `sample_id` or `id` as the identifier column. Recognized sequence columns include:

- RNA: `RNA_sequence` or `RNA_aa_code`
- peptide: `peptide_sequence`, `Protein1`, or `protein_sequence`

### Contact labels

Training requires one `{pair_id}_label.json` file per complex. Contact indices are one-based in the JSON file:

```json
{
  "contacts": [
    {"r_idx": 1, "p_idx": 1}
  ]
}
```

The current dataset loader pads RNA and peptide representations to maximum lengths of 500 nucleotides and 50 residues, respectively.

## Path configuration

The released scripts preserve the paths used in the original computing environment. Before running them, replace the hard-coded paths in the configuration blocks with paths on the local system.

| Script | Paths or settings to update |
| --- | --- |
| `predict_rna.py` | `CSV_FILE`, `FASTA_DIR`, `OUT_DIR`, `RHOFOLD_DIR` |
| `predict_peptide.py` | `CSV_FILE`, `OUT_DIR` |
| `extract_rinalmo_features.py` | `INPUT_FASTA_DIR`, `OUTPUT_DIR`, `LOCAL_MODEL_PATH` |
| `extract_esm2_features.py` | `INPUT_FASTA_DIR`, `OUTPUT_DIR`, `ESM2_PATH` |
| `extract_3d_psrt_monomer.py` | RNA and peptide PDB directories in `CONFIG`, plus the output root in `run_monomer_featurization` |
| `train.py` | `BASE_DIR` and its derived feature, label, and split directories |
| `build_3d_complex.py` | `BASE_DIR`, `MODEL_WEIGHTS`, monomer PDB directories, and output directory |

The expected derived-data layout under `BASE_DIR` is:

```text
BASE_DIR/
├── fasta/
├── Dataset_Splits/
│   ├── train_list.txt
│   ├── val_list.txt
│   └── test_list_59.txt
├── RNA_Peptide_Features/
│   ├── RNA_RiNALMo/
│   └── Peptide_ESM2/
├── features_3d/
│   └── ZHMolTopo_3D_monomer/raw_numpy/
└── RNA_Peptide_Pairwise/
    ├── labels/
    └── monomer_pdbs/
        ├── rnas/
        └── peptides/
```

## Workflow

Run the following commands from the `code` directory after updating all paths.

### 1. Predict RNA and peptide monomer structures

```bash
python predict_rna.py
python predict_peptide.py
```

If compatible monomer structures are already available, this step can be skipped. RNA structures must be named `{pair_id}_rna.pdb`, and peptide structures must be named `{pair_id}_peptide.pdb`.

### 2. Extract language-model features

```bash
python extract_rinalmo_features.py
python extract_esm2_features.py
```

The RiNALMo feature files contain the key `feat_rna`; the ESM-2 files contain `feat_peptide`.

### 3. Extract monomer PSRT descriptors

```bash
python extract_3d_psrt_monomer.py
```

For each RNA and peptide monomer, the script writes Betti, f-vector, facet, and h-vector curves. The dataset loader concatenates 13 components for each monomer over 50 filtration points, producing a 1,300-dimensional descriptor for each RNA–peptide pair.

### 4. Train the contact-prediction network

```bash
python train.py
```

The released training configuration uses AdamW, a batch size of 8, an initial learning rate of `5 × 10^-5`, cosine annealing, a positive-class weight of 20, and early stopping based on validation F1. The best checkpoint is written as `best_zhmoltoporpi_model.pth`; this historical filename is retained for compatibility with the released scripts.

### 5. Assemble three-dimensional complexes

```bash
python build_3d_complex.py
```

The script loads the pretrained contact-prediction model, selects high-ranking predicted contacts, converts them into RNA C4′–peptide Cα atom-pair restraints, and performs restraint-guided PyRosetta FastRelax. Predicted structures are written as `{pair_id}_complex.pdb`.

## Pretrained model and released results

`code/best_zhmoltoporpi_model.pth` contains the released model state dictionary. `result/mypredict_3D.zip` contains the corresponding predicted complex structures. The filenames of the model and network class are retained from development for compatibility; they refer to the ZHMolRPep contact-prediction model used in this repository.

## Reproducibility notes

- The scripts currently use configuration constants rather than command-line arguments.
- Model checkpoints for ESM-2, RiNALMo, and RhoFold are not bundled with this repository.
- PyRosetta installation and licensing are handled separately by the PyRosetta distribution.
- The ESMFold API is an external service; availability and returned models may change over time.
- Exact reproduction requires the supplied split lists, labels, derived features, monomer structures, and compatible software/model versions.

## Citation

If you use ZHMolRPep, please cite the associated manuscript. Full citation information will be added after publication.

## License

No open-source license is currently specified. A license should be added before public release to define permitted reuse and redistribution.
