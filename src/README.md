# Easy Lightning

**Easy Lightning** is a flexible and extensible framework for building deep learning models with ease using PyTorch Lightning. It simplifies training, experimentation, and deployment across tasks and datasets — with a focus on modularity and reproducibility.

It currently includes two main modules:

- **EasyTorch** — designed for vision and general deep learning tasks.
- **EasyRec** — tailored specifically for building recommendation systems.

---

## 🚀 Features

- **Modular Design**: Plug in new datasets, models, loss functions, or optimizers with minimal effort.
- **Config-Driven**: Fully customizable experiments via YAML configuration files.
- **Extendable Framework**: Add your own components — from metrics to data augmentations — without changing the core logic.
- **Built-in Modules**: Includes EasyTorch and EasyRec for general-purpose and recommendation system tasks.

---

## 📦 Installation and ⚡ QuickStart

```bash
pip install easy-lightning
```

Initialize a new project scaffold:
```bash
easy-lightning-init
```
This will create the configuration and directory structure needed to get started right away.

## 📚 Documentation

Full guides, configuration tutorials, and API references are available at:

🔗 [Easy Lightning Docs](https://easy-lightning-test.readthedocs.io/en/latest/)


## 📖 How to cite
If you use Easy Lightning in your research or project, please cite us:
```bibtex
@article{betello2024reproducible,
  title={A Reproducible Analysis of Sequential Recommender Systems},
  author={Betello, Filippo and Purificato, Antonio and Siciliano, Federico and Trappolini, Giovanni and Bacciu, Andrea and Tonellotto, Nicola and Silvestri, Fabrizio},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```
## 🤝 Contributing
We welcome contributions! If you want to add a new module or fix a bug, feel free to open an issue or submit a pull request.
