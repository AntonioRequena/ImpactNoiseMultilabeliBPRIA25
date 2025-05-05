# Code for: *Impact of Label-Level Noise on Multi-Label Learning*

This repository contains the source code and scripts needed to reproduce the experiments described in the paper:


> **"Impact of label-level noise on multi-label learning: a case study on the k-Nearest Neighbor classifier"**  
> *Antonio Requena, Antonio Javier Gallego, Jose J. Valero-Mas*  
> Proceedings of the 12th Iberian Conference on Pattern Recognition and Image Analysis (IbPRIA 2025), Coimbra, Portugal.


<br/>


This paper explores the effect of **label-level noise** in **multi-label classification**, using synthetic noise induction 
and k-Nearest Neighbor classifiers. We introduce controlled noise into multi-label datasets and evaluate the resulting performance variations.

<br/>


## 🚀 Getting Started

Follow these steps to set up the environment and reproduce the experiments.

### 1. Create a virtual environment

It's recommended to use a virtual environment to avoid dependency conflicts:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

### 2. Install dependencies

Install required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Explore the project structure

The main source files are located in the `py_projects` directory:

- `noise.py`: Contains the noise induction algorithms used in the experiments.
- `lote_auto_ruido.sh`: Bash script to run all experiments.
- `datasets/`: Folder expected to contain the multi-label datasets (e.g., emotions, scene, etc.)


### 4. Run the experiments

To execute all the experiments, run the following script:

```bash
./lote_auto_ruido.sh
```

> 💡 Make sure the script has execution permissions. You can set them with:  
> `chmod +x lote_auto_ruido.sh`

<br/>



## Citation

If you find this code useful in your research, we kindly ask you to cite the following paper:

```bibtex
@inproceedings{Requena2025,
  author    = {Antonio Requena and Antonio Javier Gallego and Jose J. Valero-Mas},
  title     = {Impact of label-level noise on multi-label learning: a case study on the k-Nearest Neighbor classifier},
  booktitle = {Proceedings of the 12th Iberian Conference on Pattern Recognition and Image Analysis (IbPRIA 2025)},
  year      = {2025},
  month     = {June 30--July 3},
  address   = {Coimbra, Portugal}
}
```
<br/>

## Acknowledgments

This work was partially funded by: 
- Generalitat Valenciana, project CIGE/2023/216.
- Spanish Ministerio de Ciencia, Innovación y Universidades, project PID2023-148259NB-I00 (LEMUR).

<br/>

---

Feel free to open an issue if you encounter any problems or have questions about the code or setup.

