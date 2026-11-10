# Datasets

The datasets of the public benchmark (`docs/BENCHMARK.md`) and of the development suite, with their creators, where
they come from and their licences. Each dataset's script in `benchmarks/public/` or `benchmarks/dev/` records what was
read at the source (file sizes, checksums, the licence text), and each manifest its licence and source.

## Public benchmark

| | Dataset | Creators | Source | Licence |
|---|---|---|---|---|
| G1 | EuroSAT (RGB) | Helber et al. (IEEE JSTARS 2019) | [Zenodo 7711810](https://zenodo.org/records/7711810), [github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT) | [MIT](https://opensource.org/license/mit); contains modified Copernicus Sentinel data |
| G2 | Caltech-101 | Li, Andreeto, Ranzato, Perona | [CaltechDATA mzrjq-6wc02](https://data.caltech.edu/records/mzrjq-6wc02) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the photos come from Google Images and keep their owners' copyright, so no image is shown |
| G3 | NCT-CRC-HE-100K and CRC-VAL-HE-7K | Kather, Halama, Marx | [10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| G4 | BloodMNIST (MedMNIST+) | Yang et al.; the images by Acevedo et al. ([10.17632/snkd93bnjr.1](https://doi.org/10.17632/snkd93bnjr.1)) | [10.5281/zenodo.10519652](https://doi.org/10.5281/zenodo.10519652) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| G5 | DeepWeeds | Olsen et al. (Scientific Reports 2019) | [github.com/AlexOlsen/DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for the images and labels |
| G6 | Beans (iBean) | AIR Lab, Makerere University | [github.com/AI-Lab-Makerere/ibean](https://github.com/AI-Lab-Makerere/ibean); the files from [huggingface.co/datasets/AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) | [MIT](https://opensource.org/license/mit) |
| G7 | CLEVR v1.0 | Johnson et al. (CVPR 2017) | [cs.stanford.edu/people/jcjohns/clevr](https://cs.stanford.edu/people/jcjohns/clevr/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| I1 | VisA | Amazon (Zou et al., ECCV 2022) | [github.com/amazon-science/spot-diff](https://github.com/amazon-science/spot-diff) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for the data |
| I2 | DAGM 2007 | Wieler, Hahn, Hamprecht | [10.5281/zenodo.12750201](https://doi.org/10.5281/zenodo.12750201) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| I3 | X-SDD | Feng et al. (Symmetry 2021) | [github.com/Fighter20092392/X-SDD-A-New-benchmark](https://github.com/Fighter20092392/X-SDD-A-New-benchmark) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), the repository's licence; the rights to the images are not documented, so no image is shown |
| I4 | RAW-FABRID | Pérez-Llorens, Albero-Albero, Silvestre-Blanes | [10.17632/db6g85xsyg.1](https://doi.org/10.17632/db6g85xsyg.1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| I5 | Wood surface defects | Kodytek, Bodzas, Bilik (F1000Research 2021) | [10.5281/zenodo.4694695](https://doi.org/10.5281/zenodo.4694695) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| I6 | Digital images of defective and good condition tyres | Pathmanaban et al. | [10.17632/bn7ch8tvyp.1](https://doi.org/10.17632/bn7ch8tvyp.1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| I7 | SDNET2018 | Maguire, Dorafshan, Thomas | [digitalcommons.usu.edu/all_datasets/48](https://digitalcommons.usu.edu/all_datasets/48/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

## Development suite

Used to develop the library (`benchmarks/dev/`), never part of the benchmark.

| Dataset | Creators | Source | Licence |
|---|---|---|---|
| Kather 2016, colorectal cancer histology textures | Kather et al. (Scientific Reports 2016) | [10.5281/zenodo.53169](https://doi.org/10.5281/zenodo.53169) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| PatchCamelyon (PCam), its validation split | Veeling et al. (MICCAI 2018), from Camelyon16 (Ehteshami Bejnordi et al., JAMA 2017) | [github.com/basveeling/pcam](https://github.com/basveeling/pcam); the files from [Zenodo 2546921](https://zenodo.org/records/2546921) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), as the authors' README gives it, following Camelyon16 |
| tf_flowers (flower_photos) | Flickr photographers, each credited in the archive's LICENSE.txt; published by TensorFlow | [flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| Water-based coated wood surface defects | Jia, Chen, Liu, Wang (Scientific Data 2026) | [Zenodo 15679291](https://zenodo.org/records/15679291) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

RP-CLASS publishes only file lists, labels (binary tasks renamed ok / nok), groups, split assignments and SHA-256 of the
images as prepared; no image. No endorsement by the authors is implied.
