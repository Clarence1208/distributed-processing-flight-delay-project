# Fiche de soutenance

## Pitch du projet

Le projet prédit, avant le départ, si un vol domestique américain arrivera en
retard. La cible finale est binaire : `is_delayed = 1` lorsque `arr_delay > 0`.
PySpark prépare et analyse un échantillon reproductible de 10 000 vols ; Python
et CatBoost entraînent le modèle sur 10 % du CSV complet ; Streamlit présente
les données, les résultats et un diagnostic de vol.

## Architecture retenue

1. **Parsing PySpark** : typage, contrôles qualité, quarantaine et Parquet.
2. **Analyse PySpark** : statistiques, groupes et corrélations sur 10 000 vols.
3. **Machine learning Python** : Polars pour la préparation, CatBoost pour la
   classification et découpage chronologique.
4. **Streamlit** : exploration, performance et formulaire de diagnostic.

Le dataset Spark et le dataset ML sont volontairement séparés. Spark répond à
l'objectif de traitement distribué et d'analyse reproductible. Le ML ne dépend
pas du Parquet Spark : il lit le CSV brut avec Python, conformément au périmètre
retenu, et utilise beaucoup plus de lignes.

## Choix importants et justifications

### Prédire tout retard, pas seulement 15 minutes

- **Choix** : un vol est en retard dès que `arr_delay > 0`.
- **Pourquoi** : c'est la définition métier demandée et la question affichée à
  l'utilisateur.
- **Conséquence** : les vols à zéro minute ou en avance sont classés à l'heure.

### Un seul objectif binaire

- **Choix** : suppression de la régression des minutes et des classifieurs de
  causes.
- **Pourquoi** : ils compliquaient le produit sans améliorer la réponse à la
  question principale. L'ancienne régression avait notamment un pouvoir
  explicatif insuffisant.
- **Conséquence** : l'artefact v6 contient un seul `CatBoostClassifier`.

### Suppression des colonnes de causes

- **Choix** : retrait de `carrier_delay`, `weather_delay`, `nas_delay`,
  `security_delay` et `late_aircraft_delay` du Parquet propre et du pipeline ML.
- **Pourquoi** : ces causes sont attribuées après le vol. Elles ne sont donc pas
  disponibles au moment de la prédiction et créeraient une fuite de données si
  elles étaient utilisées comme entrées.
- **Précision** : le code compagnie `op_unique_carrier` est conservé. Il décrit
  la compagnie prévue avant le départ ; ce n'est pas la cause `carrier_delay`.

### Exclusion des données connues après le départ

- **Choix** : `dep_delay`, `arr_delay`, les horaires réels, les temps de roulage
  et les durées réelles ne sont pas des features.
- **Pourquoi** : `dep_delay` est très corrélé à `arr_delay` (`0,966` dans
  l'analyse Spark), mais il n'est connu qu'après le départ. Son utilisation
  donnerait artificiellement un excellent score inutilisable en pratique.

### Échantillon Spark de 10 000 lignes

- **Choix** : analyse Spark sur un échantillon exact, déterministe et versionné.
- **Pourquoi** : il reste rapide sur les ordinateurs de toute l'équipe et permet
  de reproduire la démonstration sans télécharger le CSV de 1,2 Go.
- **Limite à reconnaître** : les statistiques Spark sont descriptives de cet
  échantillon, pas une preuve définitive sur tous les vols américains.

### Entraînement ML sur 10 % du CSV complet

- **Choix** : 696 596 vols achevés issus d'un échantillonnage déterministe.
- **Pourquoi** : compromis entre temps, mémoire et quantité de données. Le
  pipeline parcourt néanmoins le CSV complet pour construire les historiques et
  les volumes de planning avant l'échantillonnage.

### Découpage chronologique

- **Choix** : janvier-juillet pour l'entraînement, août pour le réglage,
  septembre-octobre pour choisir le seuil et novembre-décembre pour le test.
- **Pourquoi** : un mélange aléatoire permettrait au passé d'être évalué à
  partir d'informations apprises dans le futur. Le découpage chronologique se
  rapproche d'une vraie mise en production.
- **Conséquence** : le test n'est jamais utilisé pour choisir le seuil.

### CatBoost

- **Choix** : `CatBoostClassifier`.
- **Pourquoi** : le dataset contient beaucoup de catégories — compagnie,
  aéroports, route, numéro de vol et créneaux — ainsi que des interactions non
  linéaires. CatBoost les traite directement sans produire un énorme one-hot
  encoding.

### Features historiques sans fuite

- **Choix** : taux de retard, taux de perturbation et volumes sur des fenêtres
  de 1 à 28 jours pour le réseau, la compagnie, les aéroports et les routes.
- **Pourquoi** : le comportement récent apporte plus de contexte que les seules
  caractéristiques statiques.
- **Protection** : les fenêtres sont fermées avant le jour du vol cible ; le vol
  à prédire ne contribue donc jamais à ses propres features.

### Métriques et seuil métier

- **Choix** : priorité à la précision des alertes, avec un minimum de rappel et
  une couverture limitée.
- **Pourquoi** : un site qui annonce presque tous les vols en retard paraît peu
  fiable, même avec un F1 élevé.
- **Contrat** : borne basse de précision à 95 % ≥ 50 %, borne basse de rappel à
  95 % ≥ 20 %, couverture entre 5 et 20 %, au moins 500 alertes et stabilité
  mensuelle.
- **Pourquoi Wilson** : une valeur ponctuelle juste au-dessus de 50 % ne suffit
  pas ; l'intervalle mesure aussi l'incertitude statistique.
- **Pourquoi 20 % de couverture maximale** : avec environ un tiers de vols
  retardés, 10 % de couverture était mathématiquement trop contraignant pour
  obtenir simultanément 20 % de rappel et une précision crédible.

### Refus du seuil maximisant uniquement F1

- **Résultat** : ce seuil signalerait 66,68 % des vols avec seulement 38,66 % de
  précision.
- **Conclusion** : il augmente le rappel, mais reproduit exactement le
  comportement alarmiste rejeté par l'objectif métier.

### Streamlit protégé

- **Choix** : l'interface distingue diagnostic interne et alerte publiable.
- **Pourquoi** : le modèle ne respecte pas encore toutes les contraintes de
  stabilité. L'application ne doit pas présenter une expérience comme une
  certitude.
- **Conséquence** : la probabilité et la classe interne sont visibles, mais
  `published_delay_alert` reste vide tant que le business gate échoue ou que le
  planning journalier exact manque.

## Résultats à connaître

### Analyse PySpark

- 10 000 vols analysés ;
- 9 836 vols achevés ;
- 3 578 vols en retard, soit 36,38 % ;
- retard médian : −6 minutes ;
- 90e percentile : 47 minutes ;
- retard maximal : 2 014 minutes ;
- corrélation `dep_delay` / `arr_delay` : 0,966, mais inutilisable avant départ ;
- faibles corrélations linéaires pour les variables réellement disponibles à
  l'avance.

**Conclusion Spark** : les variables planifiées prises séparément expliquent
peu le retard. Il faut exploiter les catégories, les interactions et les
historiques, tout en évitant les données postérieures au départ.

### Modèle final sur le test novembre-décembre

| Mesure | Résultat |
|---|---:|
| Vols de test | 115 700 |
| Précision des alertes | 50,89 % |
| Borne basse de précision à 95 % | 50,19 % |
| Rappel | 26,36 % |
| Couverture | 16,97 % |
| ROC-AUC | 0,643 |

La validation septembre-octobre obtient 49,08 % de précision. En novembre, le
rappel est trop faible ; en décembre, la couverture monte à 24,22 %. Le business
gate global échoue donc malgré le passage du test global.

**Conclusion ML** : le modèle dépasse 50 % de précision sur le test global sans
signaler tous les vols, mais il n'est pas assez stable pour une publication. Il
constitue une démonstration fonctionnelle et honnête, pas un produit de
production.

## Exactitude ou précision ?

Dans la soutenance, le mot **précision** désigne la part de vrais retards parmi
les alertes émises. Ce n'est pas l'**exactitude** globale. Comme environ 67 % des
vols du test sont à l'heure, un modèle naïf prédisant toujours « à l'heure »
aurait déjà environ 67 % d'exactitude, mais ne détecterait aucun retard. C'est
pourquoi l'exactitude seule n'est pas la métrique métier retenue.

## Limites à annoncer spontanément

- une seule année de données, 2024 ;
- pas d'état opérationnel en temps réel avant le départ ;
- pas de météo prévisionnelle ni d'avis NAS connus au moment de prédire ;
- saisonnalité encore instable entre les mois ;
- annulations et déroutements exclus de la cible ;
- profils historiques arrêtés au 31 décembre 2024 ;
- besoin d'une année 2025 tenue à l'écart pour une vraie validation externe.

Ces limites ne sont pas cachées : elles expliquent pourquoi Streamlit affiche un
diagnostic expérimental.

## Déroulé de démonstration conseillé

1. Ouvrir le notebook et présenter les quatre étapes.
2. Montrer le rapport de qualité et les résultats Spark déjà exécutés.
3. Insister sur la corrélation trompeuse de `dep_delay` et la fuite de données.
4. Présenter le découpage chronologique et les principales features.
5. Montrer les résultats validation/test et expliquer la différence entre
   précision, rappel et couverture.
6. Lancer Streamlit, filtrer les vols puis ouvrir la page de performance.
7. Soumettre un vol dans le formulaire et expliquer pourquoi la classe reste un
   diagnostic non publiable.
8. Conclure sur les données supplémentaires nécessaires pour passer en
   production.

## Questions probables du jury

**Pourquoi ne pas utiliser `dep_delay`, alors que sa corrélation est excellente ?**  
Parce qu'il est connu après le départ. Ce serait une fuite de données et non une
prédiction avant départ.

**Pourquoi avoir supprimé la météo et les causes ?**  
Les colonnes présentes décrivent des causes constatées après le vol. Sans météo
prévisionnelle horodatée avant le départ, elles ne peuvent pas servir d'entrées.

**Pourquoi seulement 10 000 lignes avec Spark ?**  
Pour disposer d'une analyse distribuée rapide, déterministe et reproductible sur
tous les PC. Le ML utilise séparément 696 596 vols.

**Pourquoi ne pas annoncer que le modèle est fiable puisqu'il dépasse 50 % ?**  
Parce que ce résultat concerne le test global. La validation et la stabilité
mensuelle échouent encore ; conclure à la production serait statistiquement
incorrect.

**Comment améliorer le modèle ?**  
Ajouter l'état opérationnel réellement connu avant le départ, le planning exact,
des prévisions météo horodatées, puis valider sur une année future indépendante.

## Vérifications avant la soutenance

```bash
uv sync --extra dev --extra notebook --python 3.11
JAVA_HOME=$(/usr/libexec/java_home -v 17) uv run pytest
uv run streamlit run streamlit_app.py
```

Le notebook contient déjà ses sorties et le modèle officiel v6 est versionné.
La démonstration Streamlit ne nécessite donc pas le CSV complet ni un nouvel
entraînement.
