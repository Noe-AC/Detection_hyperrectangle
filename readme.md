2020-03-12

#Readme

*Titre :*
Détecteur de régions statistiquement peu probables.

*Contexte :*
On considère un nuage de *m* points dans un espace de dimension *n*.
Chaque point est à valeurs binaires {0,1}.

*But :*
Déterminer des hypercubes dans lesquels la proportion locale de 0 et de 1 est statistiquement éloignée de la proportion globale de 0 et de 1.

*Idée :*

1. On considère le plus petit hypercube contenant l'ensemble des *m* points.
2. On redimensionne ce dernier hypercube à un hypercube normalisé *[0,1]ⁿ*.
3. On découpe l'hypercube normalisé en hyperrectangles.

*Remarques :*
Pour l'instant, le présent code Python 3 :

1. ne fonctionne qu'en dimension *n=2*,
2. a une complexité algorithmique de *mK²ⁿ* (donc une complexité algorithmique de *mK⁴* en dimension *n=2*). Ici *K* est la résolution maximale dans chaque dimension.