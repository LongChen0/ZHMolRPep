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
│   ├── build\_3d\_complex.py
│   ├── predict\_rna.py
│   ├── predict\_peptide.py
│   ├── extract\_rinalmo\_features.py
│   ├── extract\_esm2\_features.py
│   ├── extract\_3d\_psrt\_monomer.py
│   ├── psrt.py
│   └── best\_zhmoltoporpi\_model.pth
├── dataset/
│   ├── train.zip
│   ├── val.zip
│   └── test.zip
└── result/
    └── mypredict\_3D.zip
```

The archived dataset is divided into 709 training, 39 validation, and 59 held-out test RNA–peptide complexes. `mypredict\_3D.zip` contains the predicted three-dimensional complex models.

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

* PyTorch with CUDA support
* ESM-2 3B (`esm2\_t36\_3B\_UR50D`)
* RiNALMo-Giga
* RhoFold and its pretrained checkpoint
* PyRosetta

PyRosetta is distributed separately and is not installed by the command above. The peptide-monomer script calls the public ESMFold API and therefore requires internet access.

## Input conventions

### Paired FASTA files

The language-model feature scripts expect one paired FASTA file per RNA–peptide complex. The two headers must end in `\_RNA` and `\_Peptide`:

```text
>example\_RNA
AUGCGU...
>example\_Peptide
MKR...
```

The FASTA filename, without `.fasta`, is used as the pair identifier.

### CSV input for monomer prediction

`predict\_rna.py` and `predict\_peptide.py` read `dataset\_split\_culled.csv`. The scripts accept `sample\_id` or `id` as the identifier column. Recognized sequence columns include:

* RNA: `RNA\_sequence` or `RNA\_aa\_code`
* peptide: `peptide\_sequence`, `Protein1`, or `protein\_sequence`

### Contact labels

Training requires one `{pair\_id}\_label.json` file per complex. Contact indices are one-based in the JSON file:

```json
{
  "contacts": \[
    {"r\_idx": 1, "p\_idx": 1}
  ]
}
```

The current dataset loader pads RNA and peptide representations to maximum lengths of 500 nucleotides and 50 residues, respectively.

## Path configuration

The released scripts preserve the paths used in the original computing environment. Before running them, replace the hard-coded paths in the configuration blocks with paths on the local system.

|Script|Paths or settings to update|
|-|-|
|`predict\_rna.py`|`CSV\_FILE`, `FASTA\_DIR`, `OUT\_DIR`, `RHOFOLD\_DIR`|
|`predict\_peptide.py`|`CSV\_FILE`, `OUT\_DIR`|
|`extract\_rinalmo\_features.py`|`INPUT\_FASTA\_DIR`, `OUTPUT\_DIR`, `LOCAL\_MODEL\_PATH`|
|`extract\_esm2\_features.py`|`INPUT\_FASTA\_DIR`, `OUTPUT\_DIR`, `ESM2\_PATH`|
|`extract\_3d\_psrt\_monomer.py`|RNA and peptide PDB directories in `CONFIG`, plus the output root in `run\_monomer\_featurization`|
|`train.py`|`BASE\_DIR` and its derived feature, label, and split directories|
|`build\_3d\_complex.py`|`BASE\_DIR`, `MODEL\_WEIGHTS`, monomer PDB directories, and output directory|

The expected derived-data layout under `BASE\_DIR` is:

```text
BASE\_DIR/
├── fasta/
├── Dataset\_Splits/
│   ├── train\_list.txt
│   ├── val\_list.txt
│   └── test\_list\_59.txt
├── RNA\_Peptide\_Features/
│   ├── RNA\_RiNALMo/
│   └── Peptide\_ESM2/
├── features\_3d/
│   └── ZHMolTopo\_3D\_monomer/raw\_numpy/
└── RNA\_Peptide\_Pairwise/
    ├── labels/
    └── monomer\_pdbs/
        ├── rnas/
        └── peptides/
```

## Workflow

Run the following commands from the `code` directory after updating all paths.

### 1\. Predict RNA and peptide monomer structures

```bash
python predict\_rna.py
python predict\_peptide.py
```

If compatible monomer structures are already available, this step can be skipped. RNA structures must be named `{pair\_id}\_rna.pdb`, and peptide structures must be named `{pair\_id}\_peptide.pdb`.

### 2\. Extract language-model features

```bash
python extract\_rinalmo\_features.py
python extract\_esm2\_features.py
```

The RiNALMo feature files contain the key `feat\_rna`; the ESM-2 files contain `feat\_peptide`.

### 3\. Extract monomer PSRT descriptors

```bash
python extract\_3d\_psrt\_monomer.py
```

For each RNA and peptide monomer, the script writes Betti, f-vector, facet, and h-vector curves. The dataset loader concatenates 13 components for each monomer over 50 filtration points, producing a 1,300-dimensional descriptor for each RNA–peptide pair.

### 4\. Train the contact-prediction network

```bash
python train.py
```

The released training configuration uses AdamW, a batch size of 8, an initial learning rate of `5 × 10^-5`, cosine annealing, a positive-class weight of 20, and early stopping based on validation F1. The best checkpoint is written as `best\_zhmoltoporpi\_model.pth`; this historical filename is retained for compatibility with the released scripts.

### 5\. Assemble three-dimensional complexes

```bash
python build\_3d\_complex.py
```

The script loads the pretrained contact-prediction model, selects high-ranking predicted contacts, converts them into RNA C4′–peptide Cα atom-pair restraints, and performs restraint-guided PyRosetta FastRelax. Predicted structures are written as `{pair\_id}\_complex.pdb`.

## Pretrained model and released results

`code/best\_zhmoltoporpi\_model.pth` contains the released model state dictionary. `result/mypredict\_3D.zip` contains the corresponding predicted complex structures. The filenames of the model and network class are retained from development for compatibility; they refer to the ZHMolRPep contact-prediction model used in this repository.

## Reproducibility notes

* The scripts currently use configuration constants rather than command-line arguments.
* Model checkpoints for ESM-2, RiNALMo, and RhoFold are not bundled with this repository.
* PyRosetta installation and licensing are handled separately by the PyRosetta distribution.
* The ESMFold API is an external service; availability and returned models may change over time.
* Exact reproduction requires the supplied split lists, labels, derived features, monomer structures, and compatible software/model versions.

## Citation

If you use ZHMolRPep, please cite the associated manuscript. Full citation information will be added after publication.

## License

No open-source license is currently specified. A license should be added before public release to define permitted reuse and redistribution.

