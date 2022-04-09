from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, RandomForestRegressor, RandomForestClassifier
import numpy as np
from pprint import pprint

def model_training(bow, target):
    X_train, X_test, y_train, y_test = train_test_split(bow, target, test_size=0.2, random_state=42,stratify=target)     #splits df in train, test, random_state=42 so it splits with the same way each time

    # Number of trees in random forest
    n_estimators = [int(x) for x in range(200, 2000, 200)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=1,
                                   random_state=42, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(X_train, y_train)

    print(rf_random.best_params_)

    best_random = rf_random.best_estimator_
    # evaluate(best_random, X_test, y_test)
    y_pred = best_random.predict(X_test)
    print('f1_score: ', (f1_score(y_test, (y_pred).astype(int), average='weighted')), sep='')
    print('Accuracy score: ', accuracy_score(y_test, (y_pred).astype(int)), sep='')
    print('Recall score: ', recall_score(y_test, (y_pred).astype(int), average='weighted'), sep='')
    print('Precision score: ', precision_score(y_test, (y_pred).astype(int), average='weighted'), sep='')

    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [False],
        'max_depth': [10, 40, 70],
        'max_features': ['auto'],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [3, 4, 5, 6],
        'n_estimators': [1000, 1150, 1250, 1300, 1350]
    }
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=1)

    grid_search.fit(X_train, y_train)

    best_grid = grid_search.best_estimator_

    # rf = RandomForestClassifier(n_estimators=1200, min_samples_split=5, min_samples_leaf=2, max_features='auto', max_depth=100, bootstrap=True)
    # # rf = RandomForestClassifier()
    # # from imblearn.under_sampling import RandomUnderSampler
    # #
    # # sampling_strategy = {1: 3977, 2: 2000, 3: 3000, 4:4000, 5:3000}
    # # X_train, y_train = RandomUnderSampler(sampling_strategy=sampling_strategy).fit_resample(X_train, y_train)
    # rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)

    from sklearn.metrics import confusion_matrix

    # print(confusion_matrix(y_test, y_pred))

    # evaluate(best_grid, X_test, y_test)
    y_pred = best_grid.predict(X_test)
    # print(y_pred)
    print('f1_score: ', (f1_score(y_test, (y_pred).astype(int), average='weighted')), sep='')
    print('Accuracy score: ', accuracy_score(y_test, (y_pred).astype(int)), sep='')
    print('Recall score: ', recall_score(y_test, (y_pred).astype(int), average='weighted'), sep='')
    print('Precision score: ', precision_score(y_test, (y_pred).astype(int), average='weighted'), sep='')

