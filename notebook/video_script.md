# Video script (max 30 minuten)

## Introductie (~1 min)

“In dit project heb ik machine learning modellen ontwikkeld en geoptimaliseerd om champignons te classificeren als eetbaar of giftig.”

“Het doel van dit project was om te voorspellen of een champignon eetbaar of giftig is, op basis van verschillende fysieke kenmerken uit een uitgebreide versie van de UCI Mushroom Dataset.”

“Om dit te doen heb ik een gestructureerde en modulaire aanpak gevolgd, met het opzetten van preprocessing pipelines, uitgebreide hyperparameter tuning, en een grondige evaluatie van verschillende machine learning modellen.”

“Daarnaast heb ik gekeken naar feature selectie en ensemble technieken om de modellen verder te verbeteren en te optimaliseren.”

## 1 Verkennende Data-analyse & Preprocessing (~3-4 min)

“Allereerst heb ik een verkennende data-analyse uitgevoerd om de verdeling van de features en de balans tussen eetbare en giftige paddenstoelen te begrijpen.”

“Ik heb gecontroleerd op ontbrekende waarden en inzicht gekregen in de frequentie van verschillende categorische waarden.”

“Vervolgens heb ik de data voorbereid door onderscheid te maken tussen numerieke en categorische features en hiervoor herbruikbare preprocessing pipelines gebouwd. Deze pipelines zorgen ervoor dat preprocessing consistent wordt toegepast tijdens training en validatie.”

“Door deze modulaire opzet is het eenvoudig om nieuwe modellen toe te voegen zonder steeds preprocessing-code te dupliceren.”

## 2 Helper Functies en Modulariteit (~2 min)

“Om dubbele code te vermijden en overzichtelijk te werken, heb ik helper functies ontwikkeld voor het opzetten van pipelines, het uitvoeren van hyperparameter tuning met RandomizedSearchCV en het plotten van learning curves.”

“Deze aanpak verhoogt de reproduceerbaarheid en maakt het makkelijk om verschillende modellen systematisch te trainen en te evalueren.”

## 3 Model training en tuning (~1 min)

“Ik heb een breed scala aan modellen getraind en geoptimaliseerd, waaronder: Logistic Regression met PCA, Stochastic Gradient Descent, Random Forest, Extra Trees, XGBoost en CatBoost.”

“Voor elk model heb ik parameter grids gedefinieerd om een eerste hyperparameter tuning uit te voeren met cross-validation.”

“Ook heb ik learning curves geplot om te controleren of de modellen goed generaliseren naar nieuwe data en om eventuele onder- of overfitting te detecteren.”

## 4 Model Evaluatie en Vergelijking (~4 min)

“De modellen zijn vergeleken op test accuracy, cross-validation scores en trainingstijd.”

“De beste prestaties werden behaald door tree-based ensemble modellen en gradient boosting technieken, met test accuracies boven de 99,8%.”

“Lineaire modellen scoorden duidelijk lager, wat aangeeft dat de relatie tussen features en klasse niet-lineair is.”

“De learning curves bevestigden dat de modellen goed generaliseren en geen overfitting vertonen.”

## 5 Feature Importance en Ensemble methoden (~5 min)

“Een belangrijk onderdeel was het analyseren van feature importance om te bepalen welke kenmerken het meest bijdragen aan de voorspellingen.”

“Hieruit bleek dat een kleine subset van features het merendeel van de informatie bevat.”

“Met methodes als SelectKBest op basis van mutual information heb ik geëxperimenteerd met het trainen van modellen op alleen de belangrijkste features.”

“De resultaten toonden aan dat met slechts de top 6 features de prestaties vergelijkbaar bleven met het gebruik van alle 15 features, wat leidt tot snellere en eenvoudigere modellen.”

“Ik heb ook geëxperimenteerd met ensemble technieken zoals VotingClassifier, waarbij voorspellingen van meerdere modellen worden gecombineerd.”

"Hoewel deze ensembles competitieve resultaten opleverden, was het verschil met het beste individuele model beperkt, waardoor de eenvoud van een enkel model de voorkeur kreeg.”

“Dit suggereert dat voor deze dataset goed getunede individuele modellen al optimaal presteren.”

## 6 Eindconclusie en productie (~3 min)

“Samenvattend laat dit project zien dat met een systematische aanpak van preprocessing, hyperparameter tuning en feature selectie, een zeer hoge nauwkeurigheid haalbaar is voor het classificeren van paddenstoelen.”

“Voor verdere verbetering zijn er nog enkele interessante mogelijkheden:”

“- Het samenvoegen van vergelijkbare categorieën in de data, wat de feature dimensionaliteit kan verkleinen en interpretatie kan vergemakkelijken.”

“- Verdere feature engineering, zoals het combineren van sterk gecorreleerde kenmerken.”

“- Uitbreiden van hyperparameter tuning of het proberen van alternatieve modellen zoals deep learning.”

“- Experimenteren met technieken voor het aanpakken van eventuele onbalans in de dataset, hoewel deze relatief gebalanceerd is.”

“Het uiteindelijke model, een XGBoost met geselecteerde features, biedt een uitstekende balans tussen nauwkeurigheid en efficiëntie.”

“Dit model is geëxporteerd en geïntegreerd in een FastAPI-applicatie, zodat het via een REST API real-time voorspellingen kan doen.”

"Een Docker image en Compose-bestand werden ook voorzien, voor eenvoudige DIY deployment"