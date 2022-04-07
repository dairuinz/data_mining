from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, RandomForestRegressor, RandomForestClassifier
import numpy as np
from pprint import pprint

def model_training(bow, target):
    X_train, X_test, y_train, y_test = train_test_split(bow, target, test_size=0.2, random_state=42)     #splits df in train, test, random_state=42 so it splits with the same way each time

    # answer = input('1. Regression \n2. Classification \n\nChoose 1 or 2: ')

    ######## REGRESSION ###########

    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]  # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                 'max_features': max_features,
    #                 'max_depth': max_depth,
    #                 'min_samples_split': min_samples_split,
    #                 'min_samples_leaf': min_samples_leaf,
    #                 'bootstrap': bootstrap}
    #
    # # Use the random grid to search for best hyperparameters
    # # First create the base model to tune
    # rf = RandomForestRegressor()
    # # Random search of parameters, using 3 fold cross validation,
    # # search across 100 different combinations, and use all available cores
    # rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=1,
    #                                    random_state=42, n_jobs=-1)
    # rf_random.fit(X_train, y_train) # Fit the random search model
    #
    # pprint(rf_random.best_params_)
    #
    # def evaluate(model, test_features, test_labels):
    #     predictions = model.predict(test_features)
    #     errors = abs(predictions - test_labels)
    #     mape = 100 * np.mean(errors / test_labels)
    #     accuracy = 100 - mape
    #     print('Model Performance')
    #     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    #     print('Accuracy = {:0.2f}%.'.format(accuracy))
    #
    #     return accuracy
    #
    # base_model = RandomForestRegressor(n_estimators=10, random_state=42)
    # base_model.fit(X_train, y_train)
    # # print(base_model.predict(X_test))
    # # base_accuracy = evaluate(base_model, X_train, y_train)
    # y_pred = base_model.predict(X_test)
    # print('f1_score: ', f1_score(y_test, (y_pred).astype(int), average='weighted'), sep='')
    # print('Accuracy score: ', accuracy_score(y_test, (y_pred).astype(int)), sep='')
    #
    # best_random = rf_random.best_estimator_
    # # random_accuracy = evaluate(best_random, X_train, y_train)
    # y_pred = best_random.predict(X_test)
    # print('f1_score: ', (f1_score(y_test, (y_pred).astype(int), average='weighted')), sep='')
    # print('Accuracy score: ', accuracy_score(y_test, (y_pred).astype(int)), sep='')
    #
    #
    # # print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))
    #
    # #grid search with cross validation
    #
    # # Create the parameter grid based on the results of random search
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [80, 90, 100, 110],
    #     'max_features': [2, 3],
    #     'min_samples_leaf': [3, 4, 5],
    #     'min_samples_split': [8, 10, 12],
    #     'n_estimators': [100, 200, 300, 1000]
    # }
    # # Create a based model
    # rf = RandomForestRegressor()  # Instantiate the grid search model
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
    #                             cv=3, n_jobs=-1, verbose=1)
    #
    # # Fit the grid search to the data
    # grid_search.fit(X_train, y_train)
    #
    # best_grid = grid_search.best_estimator_
    # # grid_accuracy = evaluate(best_grid, X_train, y_train)
    #
    # # print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))
    # y_pred = best_grid.predict(X_test)
    # print('f1_score: ', (f1_score(y_test, (y_pred).astype(int), average='weighted')), sep='')
    # print('Accuracy score: ', accuracy_score(y_test, (y_pred).astype(int)), sep='')



    #####CLASSIFICATION#####

    # # Number of trees in random forest
    # n_estimators = [int(x) for x in range(200, 2000, 200)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    # # pprint(random_grid)
    #
    # # Use the random grid to search for best hyperparameters
    # # First create the base model to tune
    # rf = RandomForestClassifier()
    # # Random search of parameters, using 3 fold cross validation,
    # # search across 100 different combinations, and use all available cores
    # rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=1,
    #                                random_state=42, n_jobs=-1)
    #
    # # Fit the random search model
    # rf_random.fit(X_train, y_train)
    #
    # print(rf_random.best_params_)
    #
    # best_random = rf_random.best_estimator_
    # # evaluate(best_random, X_test, y_test)
    # y_pred = best_random.predict(X_test)
    # print('f1_score: ', (f1_score(y_test, (y_pred).astype(int), average='weighted')), sep='')
    # print('Accuracy score: ', accuracy_score(y_test, (y_pred).astype(int)), sep='')
    #
    # # Create the parameter grid based on the results of random search
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [10, 15],
    #     'max_features': [2, 3],
    #     'min_samples_leaf': [3, 4, 5, 6],
    #     'min_samples_split': [3, 4, 5, 6],
    #     'n_estimators': [1150, 1200, 1250, 1300, 1350]
    # }
    # # Create a based model
    # rf = RandomForestClassifier()
    # # Instantiate the grid search model
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
    #                                cv=3, n_jobs=-1, verbose=1)
    #
    # grid_search.fit(X_train, y_train)
    #
    # best_grid = grid_search.best_estimator_

    #alvanikos tropos roustai
    rf = RandomForestClassifier(n_estimators=140, min_samples_split=3, min_samples_leaf=2, max_features='auto', max_depth=5, bootstrap=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # evaluate(best_grid, X_test, y_test)
    # y_pred = best_grid.predict(X_test)

    print('f1_score: ', (f1_score(y_test, (y_pred).astype(int), average='weighted')), sep='')
    print('Accuracy score: ', accuracy_score(y_test, (y_pred).astype(int)), sep='')

