# Video script (max 30 minuten)

## Introductie (~1 min)

In dit project heb ik gewerkt aan een machine learning model om champignons te classificeren als eetbaar of giftig.

Het idee is om op basis van verschillende fysieke kenmerken — uit een uitgebreide versie van de UCI Mushroom Dataset — te voorspellen of je de champignon veilig kunt eten.

Om dit voor elkaar te krijgen, heb ik een gestructureerde aanpak gevolgd. Denk aan het bouwen van preprocessing pipelines, uitgebreid hyperparameter tuning doen, en verschillende modellen vergelijken.

Ik heb ook gekeken hoe ik de prestaties kon verbeteren met feature selectie en ensemble technieken.

## 1 Verkennende Data-analyse & Preprocessing (~3-4 min)

Ik ben begonnen met een verkennende data-analyse. Daarmee krijg je gevoel voor hoe de features verdeeld zijn en hoe de verhouding is tussen eetbare en giftige paddenstoelen.

Ik heb gecontroleerd of er ontbrekende waarden waren en gekeken naar hoe vaak bepaalde categorische waarden voorkomen.

Daarna heb ik de data voorbereid. Daarbij heb ik aparte preprocessing pipelines gemaakt voor numerieke en categorische features. Het voordeel hiervan is dat preprocessing altijd consistent gebeurt, zowel bij training als bij validatie.

Met deze modulaire aanpak kun je bovendien makkelijk nieuwe modellen toevoegen, zonder dat je preprocessing telkens opnieuw hoeft te schrijven.

## 2 Helper Functies en Modulariteit (~2 min)

Om te voorkomen dat ik overal dezelfde stukken code moest kopiëren, heb ik een paar handige helper functies geschreven. Daarmee kan ik snel pipelines opzetten, hyperparameter tuning uitvoeren met RandomizedSearchCV, en learning curves plotten.

Die aanpak maakt het hele proces een stuk overzichtelijker en zorgt ervoor dat ik modellen makkelijk en op een consistente manier kan trainen en evalueren.

## 3 Model training en tuning (~1 min)

Ik heb verschillende modellen getraind en geoptimaliseerd. Denk aan Logistic Regression met PCA, Stochastic Gradient Descent, Random Forest, Extra Trees, XGBoost en CatBoost.

Voor elk model heb ik met cross-validation en hyperparameter tuning gekeken welke instellingen het beste werken.

Ook heb ik learning curves geplot om te controleren of de modellen goed generaliseren naar nieuwe data en om te zien of er sprake is van over- of underfitting.

## 4 Model Evaluatie en Vergelijking (~4 min)

Ik heb de modellen met elkaar vergeleken op test accuracy, cross-validation scores en trainingstijd.

De tree-based ensemble modellen en gradient boosting technieken presteerden het beste, met test accuracies boven de 99,8%.

Lineaire modellen scoorden duidelijk lager. Dat geeft aan dat de relatie tussen de features en de klasse niet lineair is.

De learning curves lieten zien dat de modellen goed generaliseren en geen overfitting vertonen.

## 5 Feature Importance en Ensemble methoden (~5 min)

Ik heb ook gekeken naar feature importance, om te bepalen welke kenmerken het meest bijdragen aan de voorspellingen.

Uiteindelijk bleek dat een kleine subset van features het meeste informatie bevat.

Daarom heb ik geëxperimenteerd met SelectKBest, waarbij ik alleen de belangrijkste features gebruikte om nieuwe modellen te trainen.

De resultaten waren verrassend goed. Met slechts de top 6 features bleef de performance vrijwel gelijk aan het gebruik van alle 15 features. Dit levert snellere en eenvoudiger modellen op.

Bovendien heb ik gekeken naar ensemble technieken, zoals de VotingClassifier. Hierbij combineer je de voorspellingen van meerdere modellen.

De ensembles presteerden prima, maar het verschil met het beste individuele model was beperkt. Daarom koos ik uiteindelijk voor het eenvoudiger XGBoost model.

Bij deze dataset lijken goed getunede individuele modellen dus al optimaal te presteren.

## 6 Eindconclusie en productie (~3 min)

Samenvattend: met een systematische aanpak van preprocessing, hyperparameter tuning en feature selectie kun je deze paddenstoelen heel nauwkeurig classificeren.

Er zijn natuurlijk nog verbeterpunten. Ik zou bijvoorbeeld vergelijkbare categorieën in de data kunnen samenvoegen om het model compacter te maken.

Ook kan ik verder kijken naar feature engineering, of alternatieve modellen zoals deep learning proberen.

Een ander idee is om technieken toe te passen die eventuele onbalans in de dataset aanpakken, al is deze dataset al redelijk gebalanceerd.

Het uiteindelijke model dat ik nu gebruik — een XGBoost met geselecteerde features — biedt een goede balans tussen nauwkeurigheid en efficiëntie.

Ik heb dit model geëxporteerd en geïntegreerd in een FastAPI-applicatie, zodat je via een REST API real-time voorspellingen kunt opvragen.

Tot slot heb ik ook een Docker image en Compose-bestand gemaakt, zodat je het hele project makkelijk zelf kunt deployen.