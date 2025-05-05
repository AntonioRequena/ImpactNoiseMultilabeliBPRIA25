

# Multi-label Noise Induction Experiments

This repository contains the source code required to reproduce the experiments described in the related article.

## Getting Started

To set up the environment and run the experiments, follow the steps below:

### 1. Create a virtual environment

It's recommended to use a virtual environment to avoid dependency conflicts:
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

### 2. Install required dependencies

All dependencies are listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Explore the project structure

The main source files are located in the `py_projects` directory:

- `noise.py`: Contains the noise induction algorithms used in the experiments.

### 4. Run the experiments

To execute all the experiments automatically, run the following script:
```bash
./lote_auto_ruido.sh
```

> 💡 Make sure the script has execution permissions. You can set them with:  
> `chmod +x lote_auto_ruido.sh`

---

Feel free to open an issue if you encounter any problems or have questions about the code or setup.



### Citations

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


### Acknowledgments

This work was partially funded by the Generalitat Valenciana through project CIGE/2023/216 and 
the Spanish Ministerio de Ciencia, Innovación y Universidades through project PID2023-148259NB-I00 (LEMUR).



