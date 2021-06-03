# SPH_Projet

Ce projet fait partie du cours Smoothed Particle Hydrodynamics. L'objectif est de réaliser des simulations numériques à l'aide de cette méthode tout en intégrant des parois mobiles. Pour ce faire, on utilise l'article "Nonlinear water wave interaction with floating bodies in SPH" de B.Bouscasse et al. disponible dans le dossier.

Les objectifs fixés sont présentés ci-dessous et sont de difficulté croissante : 

-1 : Etude d'un l'écoulement de couette plan.

-2 : Etude d'un écoulement dans une cavité entrainée.

-3 : Etude d'un écoulement suite à la chute d'un solide.


Afin de simuler ces différents écoulements nous avons créé de nouvelles fonctions dans les fichiers Python existants. 

Celles-ci peuvent être aisément identifiées à l'aide du commentaire initial : 
#-------------------------------------------------------------------------------------------
#---------------------------------Project function------------------------------------------
#-------------------------------------------------------------------------------------------


Le fichier solidStuffManagement.py est dédié à la résolution du troisième objectif. 
Toutes les fonctions qui y sont présentes sont nouvelles. 