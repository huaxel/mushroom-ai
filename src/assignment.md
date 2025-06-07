# Assignment

## Minimum Requirements

You will need to train at least 3 different models on the data set. Make sure to include the reason for your choice (e.g., for dealing with categorical features).

- Define the problem, analyze the data, and prepare the data for your model.
- Train at least 3 models (e.g., decision trees, nearest neighbour, ...) to predict whether a mushroom is poisonous or edible.
  - You are allowed to use any machine learning model from scikit-learn or other methods, as long as you motivate your choice.
- For each model, optimize the model parameters settings (tree depth, hidden nodes/decay, number of neighbours, ...).
  - Show which parameter setting gives the best model.
- Compare the best parameter settings for the models and estimate their errors on unseen data.
  - Investigate the learning process critically (overfitting/underfitting). Can you show that one of the models performs better?

All results, plots, and code should be handed in as an interactive [iPython notebook](http://ipython.org/notebook.html). Simply providing code and plots does not suffice; you are expected to accompany each technical section with explanations and discussions on your choices/results/observations, etc., in the notebook and in a video (by recording your screen and voice).

**The deadline for the notebook is 15/06/2025.**  
**The deadline for the video is 19/06/2025.**

## Optional Extensions

You are encouraged to try and see if you can further improve on the models you obtained above.

This is not necessary to obtain a good grade on the assignment, but any extensions on the minimum requirements will count for extra credit.

Some suggested possibilities to extend your approach are:

- Build and host an API for your best performing model.
  - You can create an API using Python frameworks such as FastAPI, Flask, ...
  - You can host an API for free on Render, using your student credit on Azure, ...
- Try to combine multiple models.
  - Ensemble and boosting methods try to combine the predictions of many simple models.
  - This typically works best with models that make different errors.
  - Scikit-learn has some support for this; [see here](http://scikit-learn.org/stable/modules/ensemble.html).
  - You can also try to combine the predictions of multiple models manually, i.e., train multiple models and average their predictions.
- Investigate whether all features are necessary to produce a good model.
  - Feel free to look up additional resources and papers to find more information; see e.g., [sklearn feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) for the feature selection module provided by the scikit-learn library.

## Additional Remarks

- Depending on the model used, you may want to [scale](http://scikit-learn.org/stable/modules/preprocessing.html) or [encode](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) your (categorical) features X and/or outputs y.
- Refer to the [SciPy](http://scipy.org/docs.html) and [Scikit-learn](http://scikit-learn.org/stable/documentation.html) documentations for more information on classifiers and data handling.
- You are allowed to use additional libraries, but provide references for these.
- The assignment is **individual**. All results should be your own. Plagiarism will not be tolerated.
