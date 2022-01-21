# AI4Indistry - Use case PRODITEC

## Architecture utilisée

Nous utilisons l'architecure ResNet50 V2.
Nous considérons le problème comme un problème de régression. C'est-à-dire que les classes sont représentées par un entier entre 0 et 4. Le modèle prédit une valeur réelle que l'on arondit pour obtenir la classe d'une image.

## Exécuter le code

Pour lancer le code, il faut d'abord lancer `python Preprod.py` puis `python Model_final.py`.