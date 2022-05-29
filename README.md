# FYS4411 - Computational Physics II: Quantum Mechanical Systems

## Project 1:

This repository contains programs, material and report for project 1 in FYS4411 made in collaboration between [Jørn Eirik](https://github.com/JornEirikBetten), [Nicolai](https://github.com/nicolossus) and [Aleksandar](https://github.com/aleksda).

In this project, we build a variational Monte Carlo method to estimate the ground state energy of an ultracold, dilute Bose gas in harmonic traps. We use a trial wave function composed of a single particle Gaussian with a single variational parameter and a hard sphere Jastrow factor for pair correlations. We consider two Markov chain Monte Carlo sampling algorithms; a random walk Metropolis with Gaussian proposals and a Langevin Metropolis-Hastings with proposals driven according to the gradient flow of the probability density of the trial wave function. The methods are implemented in a Python framework with automatic differentiation, procedures for tuning scale parameters and gradient descent optimizations of the variational parameter. The blocking method, which accounts for the autocorrelation of a Markov chain, is used to calculate the statistical error of the variational energy. Our `vmc` package can be found in the [src directory](https://github.com/nicolossus/FYS4411-Project1/tree/main/src).

## Contents

The [latex directory](https://github.com/nicolossus/FYS4411-Project1/tree/main/latex) contains the LaTeX source for building the report, as well as [figures](https://github.com/nicolossus/FYS4411-Project1/tree/main/latex/figures) and [tables](https://github.com/nicolossus/FYS4411-Project1/tree/main/tables) generated in the analyses.

The [notebooks directory](https://github.com/nicolossus/FYS4411-Project1/tree/main/notebooks) contains Jupyter Notebooks used in the analyses.

The [report directory](https://github.com/nicolossus/FYS4411-Project1/tree/main/report) contains the report rendered to PDF from the LaTeX source.

The [resources directory](https://github.com/nicolossus/FYS4411-Project1/tree/main/resources) contains project resources such as literature.

The [src directory](https://github.com/nicolossus/FYS4411-Project1/tree/main/src) contains the source code of the `vmc` package.

The [tests folder](https://github.com/FYS4411-Project1/tree/main/tests) contains unit tests. Run tests locally with `pytest`:

    $ pytest tests -v -W ignore

## Virtual environment
`environment.yml` contains the dependencies of the `vmc` package, including `JAX` used for automatic differentiation. Install the `conda` environment:

    $ conda env create --file environment.yml

To activate the environment:

    $ conda activate bios1100

To deactivate the environment:

    $ conda deactivate
