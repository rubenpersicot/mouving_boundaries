# mouving_boundaries

The project [mouving_boundaries](https://github.com/rubenpersicot/mouving_boundaries) is part of the course "Smoothed Particle Hydrodynamics" (SPH) taught at the [Ecole des Ponts Paristech](https://www.ecoledesponts.fr/) by Damien Violeau and Remi Carmigniani. 

### 1. Objectives
Our goal is to undertake numerical simulations using the SPH method on configurations in which boundaries are mobile. In order to do so, we based our work on the articles available on the "doc" folder.

Here are our objectives, listed by increasing difficulty :

-1 : Study of a [Couette's flow](https://github.com/rubenpersicot/mouving_boundaries/blob/main/1Couette_flow.ipynb).

-2 : Study of a [lid driven cavity](https://github.com/rubenpersicot/mouving_boundaries/blob/main/2Lid_driven_cavity.ipynb).

-3 : Study of a flow induced by a [floating solid](https://github.com/rubenpersicot/mouving_boundaries/blob/main/3Floating_solid.ipynb).

Each objective is adressed in a jupyter notebook named after the objective.
Python modules used to solve these problems are located in the src project. Note a major part of the code in this module was written by Dr Carmigniani. More precisely, it is a sum up of all the practical exercises we did in class. 

Nonetheless, to simulate these flows we had to create new functions. They can be easily witnessed by the following comment :
```python
# Project function
```
The module [solidStuffManagement.py](https://github.com/rubenpersicot/mouving_boundaries/blob/main/src/solidStuffManagement.py) is dedicated to solve the third objective. Every function located in it is new.

### 2. Repository architecture 
- [figures](https://github.com/rubenpersicot/mouving_boundaries/tree/main/figures) contains figures.
- [references](https://github.com/rubenpersicot/mouving_boundaries/tree/main/references) contains all the articles we read to do this project.
- [results](https://github.com/rubenpersicot/mouving_boundaries/tree/main/results) contains the three simulations. Each simulation is a set of frames. 
- [src](https://github.com/rubenpersicot/mouving_boundaries/tree/main/src) contains Python modules.



## Installation

To check our project, you will need [anaconda](https://docs.anaconda.com/anaconda/install/index.html).

## Usage 
Simply run the jupyter notebooks !

## Final grade
17/20

## Credits 
[Damien Violeau](https://www.ecoledesponts.fr/damien-violeau), [Remi Carmigniani](https://github.com/remingtonCarmi), [Yohan Lanier](https://github.com/yohan-lanier)

## License
[MIT](https://choosealicense.com/licenses/mit/)
