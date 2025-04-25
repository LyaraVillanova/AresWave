![banner](https://github.com/user-attachments/assets/555ebae5-0d10-47dd-a00e-19ed2b748dd7)

AresWave is a Python software package for calculating optimal solutions of depth, moment tensor, and 1D velocity models using Particle Swarm Optimization (PSO) as the methodology, and DSMpy to generate synthetic seismograms.

All main dependencies are listed below:



Título
# AresWave

**Estimativa de parâmetros fonte de marsquakes via ajuste de formas de onda com otimização estocástica**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

O AresWave é um pacote Python para a estimativa de profundidade e mecanismo focal de eventos sísmicos marcianos (marsquakes), combinando modelagem de forma de onda (via DSMpy) com otimização estocástica por Particle Swarm Optimization (PSO). O código também inclui ferramentas para ajuste de modelos 1D e estimativas bayesianas de profundidade com base em tempos S–P.
Funcionalidades
## ✨ Funcionalidades

- Geração de formas de onda sintéticas com DSMpy
- Comparação de formas de onda reais e sintéticas (P e S)
- Otimização de parâmetros fonte com PSO
- Estimativa bayesiana de profundidade via PyMC
- Inversão iterativa de modelos de velocidade 1D
- Foco no evento S0185a (SEIS/InSight – Marte)
Instalação
## 📦 Instalação

Clone o repositório:

```
git clone https://github.com/seuusuario/areswave.git
cd areswave
```

Crie e ative um ambiente virtual (recomendado):

```
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

Instale as dependências:

```
pip install -r requirements.txt
```
Requisitos
## 🧪 Requisitos

AresWave requer as seguintes bibliotecas:

```
numpy
scipy
matplotlib
obspy
pymc
arviz
DSMpy
```

> Recomenda-se Python 3.8 ou superior. Certifique-se de que o DSMpy está corretamente instalado a partir do repositório oficial: https://github.com/GeodynamicWorld/DSMpy
Como usar
## 🚀 Como usar

```python
from areswave import optimize_source

# Defina o caminho dos dados reais e o modelo inicial
optimize_source(event='S0185a', model='dsm_mars.nd', components=['Z', 'R', 'T'])
```

Scripts de exemplo e notebooks estão disponíveis na pasta `examples/`.
Resultados
## 📊 Resultados

O método foi aplicado com sucesso ao evento S0185a, obtendo uma profundidade de ~39 km e mecanismo focal normal. Veja detalhes no artigo (link abaixo).
Publicação
## 📄 Publicação

Se usar este código, por favor cite:

> [Seu Nome], 2025. *AresWave: Estimation of marsquake source parameters by waveform fitting with stochastic optimization*. [Link para o preprint ou DOI]
Contato
## 📫 Contato

Para dúvidas ou colaborações, envie um e-mail para: [seu.email@instituicao.edu]
Licença
## 🪐 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
