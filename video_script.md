# Video script

## 1 Introductie (~10-20 sec)

“Hallo, in deze video presenteer ik mijn project voor de Mushroom Classification opdracht.”

“Het doel was om te voorspellen of een champignon eetbaar of giftig is, op basis van verschillende fysieke kenmerken.”

“Ik heb hiervoor een gestructureerde aanpak gebruikt, met modulaire pipelines, hyperparameter tuning en een kritische evaluatie van de modellen.”

## 2 EDA & preprocessing (~20-30 sec)

“Allereerst heb ik een verkennende data-analyse uitgevoerd om de verdeling van de features en de balans tussen de klassen te begrijpen.”

“Daarna heb ik de data voorbereid door numerieke en categorische features te identificeren, en herbruikbare preprocessing pipelines op te bouwen.”

## 3 Helper Functies (~10-15 sec)

“Om dubbele code te vermijden en consistent te werken, heb ik herbruikbare functies geïmplementeerd voor het opbouwen van pipelines, hyperparameter tuning met GridSearchCV, en het plotten van learning curves.”

## 4 Model training en tuning (~1 min)

“Ik heb de volgende modellen getraind en geoptimaliseerd:”

- Logistic Regression (met PCA),
- Support Vector Classifier,
- K-Nearest Neighbors,
- Random Forest,
- Extra Trees,
- XGBoost,
- CatBoost,
- LightGBM.

“Voor elk model heb ik parameter grids gedefinieerd en optimalisatie uitgevoerd met cross-validation.”

“Daarnaast heb ik learning curves geplot om de generalisatieprestaties van de modellen te analyseren.”

## 5 Deep tuning (~20-30 sec)

“Voor de best presterende modellen — Random Forest, XGBoost, LightGBM en CatBoost — heb ik een extra deep hyperparameter tuning uitgevoerd met uitgebreidere grids om de prestaties verder te verbeteren.”

“Hiermee kon ik de resultaten nog verder optimaliseren.”

## 6 Model evaluatie en vergelijking (~20-30 sec)

“Ik heb de modellen vergeleken door test accuracy, generalisatieprestaties en feature importances te plotten.”

“Het best presterende model was [] met een test accuracy van X%.”

“De learning curves bevestigden dat dit model goed generaliseert naar nieuwe data.”

## 7 Extra analyse (~30-45 sec)

“Ik heb ook onderzocht of alle features noodzakelijk waren. Door feature importance analyse, SelectKBest en RFECV heb ik vastgesteld dat een gereduceerde feature set vergelijkbare prestaties kan behalen.”

“Daarnaast heb ik modellen gecombineerd met zowel StackingClassifier als VotingClassifier. Dit liet zien dat ensemble technieken competitieve resultaten kunnen opleveren, al presteerden ze in dit geval niet significant beter dan het beste individuele model.”

## 8 Eindconclusie (~20-30 sec)

“Samenvattend heb ik aangetoond dat met een goede preprocessing, optimalisatie en kritische evaluatie, een hoge nauwkeurigheid haalbaar is voor deze classificatietaak.”

“Het uiteindelijke model is geëxporteerd om gebruikt te worden in een REST API, als optionele uitbreiding van dit project.”

“Dankjewel voor het kijken"
