# TP 2 — Deep Learning : Étude et comparaison de modèles de vision par ordinateur sur MNIST

**Réalisé par :** Abouyahia Kenza  
**Formation :** Master 2 — Sécurité IT & Big Data  
**Établissement :** Faculté des Sciences et Techniques de Tanger  
**Université :** Abdelmalek Essaâdi  
**Module :** Deep Learning  
**Encadrant :** Pr. Lotfi ELAACHAK  
**Année universitaire :** 2026–2027

---

## 1. Introduction

Dans le cadre du module Deep Learning, j'ai réalisé ce deuxième travail pratique afin de découvrir et de comparer plusieurs architectures de réseaux de neurones utilisées en vision par ordinateur.

L'objectif était de comprendre comment ces modèles fonctionnent, comment les entraîner avec PyTorch et surtout d'observer leurs performances sur un même problème de classification.

Pour cela, j'ai travaillé avec le dataset MNIST, qui contient des images de chiffres manuscrits allant de 0 à 9. Le TP porte sur cinq architectures : un CNN classique, Faster R-CNN, VGG16, AlexNet et un Vision Transformer (ViT) construit à partir de zéro.

Les expérimentations ont été réalisées sur Google Colab avec une accélération GPU. Pour les modèles les plus coûteux, j'ai utilisé un mode d'exécution rapide afin de limiter le temps de calcul.

## 2. Préparation des données

Le dataset MNIST contient 70 000 images de chiffres manuscrits : 60 000 images d'apprentissage et 10 000 images de test. Chaque image est en niveaux de gris, avec une résolution de 28 × 28 pixels.

Avant l'entraînement, les images ont été préparées et normalisées. Certaines architectures, notamment VGG16, AlexNet et Faster R-CNN, nécessitent aussi une adaptation des images d'entrée, car elles ont été conçues pour traiter des images RGB.

Les modèles n'ont pas tous utilisé la même quantité de données d'entraînement. Cette différence est importante pour interpréter correctement les résultats.

## 3. Partie 1 — Classification avec différentes architectures

### 3.1. CNN classique

J'ai commencé par construire un réseau de neurones convolutif avec PyTorch. L'architecture comprend deux couches de convolution suivies d'activations ReLU et de Max Pooling. Les caractéristiques extraites sont ensuite transmises à des couches entièrement connectées pour effectuer la classification des dix chiffres. Une couche Dropout permet de réduire le risque de surapprentissage.

Les principaux paramètres choisis sont : optimiseur Adam, learning rate de 0,001, batch size de 128, 10 époques, Dropout de 0,5 et fonction de perte CrossEntropyLoss. Le modèle a été entraîné sur GPU Tesla T4.

| Métrique | Valeur |
|---|---:|
| Accuracy | **99,20 %** |
| F1-score macro | 99,19 % |
| Loss test | 0,025146 |
| Temps d'entraînement | 152,15 s |

Les résultats sont très satisfaisants : le CNN reconnaît correctement la grande majorité des chiffres manuscrits. La loss d'entraînement a diminué progressivement et le meilleur modèle a été obtenu à l'époque 9. La matrice de confusion montre quelques erreurs entre des chiffres ayant des formes proches, mais elles restent peu nombreuses.

Cette première expérience m'a permis de mieux comprendre l'intérêt des convolutions pour extraire automatiquement les caractéristiques d'une image.

### 3.2. Faster R-CNN

J'ai ensuite expérimenté Faster R-CNN, une architecture principalement utilisée pour la détection d'objets. Contrairement au CNN classique, ce modèle cherche à identifier un objet et à déterminer sa position dans l'image.

Pour l'adapter à MNIST, chaque image a été associée à une boîte englobante et à une étiquette correspondant au chiffre représenté. Cette adaptation reste simplifiée, puisque les images MNIST contiennent généralement un seul chiffre.

| Métrique | Valeur |
|---|---:|
| Accuracy | 18,84 % |
| F1-score macro | 17,45 % |
| Temps d'entraînement | 365,18 s |

Les performances sont nettement inférieures à celles du CNN. Une grande partie des images de test n'a pas produit de détection retenue par le modèle ; ces cas ont été comptés comme des erreurs de classification.

Cette expérience m'a montré qu'une architecture plus complexe n'est pas nécessairement plus efficace pour tous les problèmes. Faster R-CNN est surtout intéressant lorsqu'on souhaite détecter et localiser des objets ; pour reconnaître un chiffre isolé, un CNN classique est plus adapté.

### 3.3. VGG16

Pour cette expérience, j'ai utilisé VGG16, un réseau convolutif préentraîné sur ImageNet. L'objectif était de découvrir le Transfer Learning : réutiliser les caractéristiques apprises par un modèle sur un autre jeu de données. La partie classification a été adaptée aux dix classes de MNIST.

| Métrique | Valeur |
|---|---:|
| Accuracy | 97,78 % |
| F1-score macro | 97,76 % |
| Temps d'entraînement | 101,88 s |

VGG16 obtient de très bons résultats malgré le nombre limité d'images utilisées. Cela montre l'intérêt du transfert d'apprentissage lorsqu'on dispose de peu de données ou que l'on souhaite réduire le coût d'entraînement. Le modèle reste toutefois relativement volumineux par rapport à un CNN simple.

### 3.4. AlexNet

J'ai également testé AlexNet, une autre architecture convolutive préentraînée sur ImageNet, adaptée pour reconnaître les dix chiffres de MNIST.

| Métrique | Valeur |
|---|---:|
| Accuracy | 95,22 % |
| F1-score macro | 95,14 % |
| Temps d'entraînement | 45,61 s |

AlexNet obtient une accuracy légèrement inférieure à VGG16, mais son entraînement a été plus rapide dans cette expérience. Cette comparaison illustre le compromis entre la complexité d'un modèle, ses performances et les ressources nécessaires.

## 4. Partie 2 — Vision Transformer (ViT)

La deuxième partie du TP est consacrée aux Vision Transformers. Contrairement aux CNN, qui reposent principalement sur des convolutions, les ViT utilisent le mécanisme de Self-Attention.

J'ai construit un ViT à partir de zéro avec PyTorch, sans poids préentraînés. L'image est découpée en patches de 7 × 7 pixels, transformés en embeddings. Des encodages positionnels et des couches Transformer permettent ensuite d'apprendre les relations entre les différentes parties de l'image.

| Métrique | Valeur |
|---|---:|
| Accuracy | 76,64 % |
| F1-score macro | 76,52 % |
| Temps d'entraînement | 10,05 s |

Le ViT obtient des performances inférieures aux CNN dans cette expérience. Cependant, il a été entraîné seulement pendant trois époques sur 6 000 images. Ce résultat ne signifie donc pas que les Vision Transformers sont moins performants en général. Un entraînement plus long, davantage de données et un ajustement des hyperparamètres pourraient améliorer ses résultats.

Cette partie m'a surtout permis de mieux comprendre les patches, les embeddings et le mécanisme de Self-Attention.

## 5. Comparaison générale des modèles

| Modèle | Accuracy | F1-score macro | Temps (s) |
|---|---:|---:|---:|
| CNN | **99,20 %** | **99,19 %** | 152,15 |
| Faster R-CNN | 18,84 % | 17,45 % | 365,18 |
| VGG16 | 97,78 % | 97,76 % | 101,88 |
| AlexNet | 95,22 % | 95,14 % | 45,61 |
| ViT | 76,64 % | 76,52 % | 10,05 |

Dans cette expérimentation, le CNN classique obtient la meilleure accuracy, suivi de VGG16 et d'AlexNet. Le Transfer Learning a permis aux deux modèles préentraînés d'obtenir de bonnes performances avec un nombre réduit d'images.

Faster R-CNN présente les résultats les plus faibles, principalement en raison de son adaptation à une tâche pour laquelle il n'a pas été conçu initialement. Le ViT montre les limites d'un entraînement très court sans préentraînement.

Cette comparaison reste expérimentale : le CNN a été entraîné sur 54 000 images, Faster R-CNN sur 2 000, et VGG16, AlexNet et ViT sur 6 000 chacun. Les performances et les durées ne sont donc pas comparables à conditions strictement identiques. Les fonctions de perte de détection et de classification ne sont pas directement comparables non plus.

## 6. Conclusion générale

Ce travail pratique m'a permis d'explorer plusieurs approches de Deep Learning appliquées à la vision par ordinateur. Les résultats montrent que le choix d'une architecture dépend avant tout du problème à résoudre.

Pour une tâche relativement simple comme la reconnaissance des chiffres manuscrits, le CNN classique a donné les meilleurs résultats dans mes expérimentations. VGG16 et AlexNet ont montré l'intérêt du Transfer Learning, tandis que Faster R-CNN m'a permis de découvrir les particularités de la détection d'objets.

Enfin, la construction du Vision Transformer a été une occasion de découvrir une approche différente, fondée sur les mécanismes d'attention. Même si certains modèles ont obtenu des performances plus faibles, leurs résultats restent utiles pour comprendre leurs limites et les conditions nécessaires à un bon apprentissage.

## 7. Synthèse personnelle

Ce TP m'a permis de mettre en pratique plusieurs notions de Deep Learning étudiées en cours.

Au début, je connaissais surtout le fonctionnement général des réseaux de neurones convolutifs. En réalisant les différentes expérimentations, j'ai mieux compris comment les architectures se distinguent et pourquoi elles ne donnent pas toutes les mêmes résultats sur un même dataset.

J'ai notamment appris à préparer les données avec PyTorch, à définir les paramètres d'entraînement, à utiliser le GPU et à analyser les résultats à partir de métriques comme l'accuracy, le F1-score et la loss.

L'une des choses qui m'a le plus marquée est que le modèle le plus complexe n'est pas forcément le meilleur. Faster R-CNN, par exemple, est une architecture puissante, mais son utilisation sur MNIST n'a pas donné de bons résultats dans cette configuration.

J'ai également découvert l'intérêt du Transfer Learning avec VGG16 et AlexNet : il est possible de réutiliser un modèle déjà entraîné au lieu de repartir systématiquement de zéro.

La partie Vision Transformer était un peu différente, notamment avec les notions de patches et de Self-Attention. Même si les résultats sont restés modestes, elle m'a permis de mieux comprendre cette architecture.

Enfin, j'ai retenu qu'il ne suffit pas d'obtenir une bonne accuracy pour juger un modèle. Il faut aussi tenir compte du temps d'entraînement, de la quantité de données utilisées, des ressources disponibles et de la nature du problème.

Ce TP m'a donc permis de progresser aussi bien sur le plan pratique que sur l'analyse des résultats.

---

**Abouyahia Kenza**  
Master 2 — Sécurité IT & Big Data  
FST Tanger — Université Abdelmalek Essaâdi  
Année universitaire 2026–2027
