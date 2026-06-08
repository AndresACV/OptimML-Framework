# Appendix of Paper: A framework for hyperparameter configuration inthe context of software defect density prediction using bio-inspired optimization algorithms.

## Table 1: Data set description based on Lamba et al. (2017)

| Dataset | Attributes | Observations | Description |
|---------|------------|--------------|-------------|
| ant-1 | 21 | 745 | Java library and command-line tool for building applications (Apache Ant). |
| poi-2 | 21 | 442 | Java APIs for manipulating Microsoft Office file formats (Apache POI). |
| camel-1 | 21 | 965 | Integration framework for Java applications (Apache Camel). |
| tomcat | 21 | 858 | Open-source implementation of Java Servlet and JSP specifications (Apache Tomcat). |
| ivy-2 | 21 | 352 | Dependency management tool integrated with Apache Ant. |
| velocity-1 | 21 | 229 | Java-based template engine for web applications (Apache Velocity). |
| jedit-4 | 21 | 492 | Text editor for programmers with plugin architecture. |
| xalan-2 | 21 | 885 | XSLT processor for transforming XML documents (Apache Xalan). |
| log4j-1 | 21 | 135 | Logging utility for Java applications (Apache Log4j). |
| xerces-1 | 21 | 437 | XML parser and toolkit (Apache Xerces). |

## Table 2: Comparison of Bio-inspired approaches with Lasso algorithm

| Dataset | Approach | RMSE | Time (s) | Hyperparameters |
|---------|----------|------|----------|-----------------|
| ant-1 | Genetic | 0.493 | 0.33 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1615, "clf__tol": 0.00033822480140215046, "clf__selection": "cyclic"} |
| ant-1 | PSO | 0.493 | 0.34 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| ant-1 | CDEA | 0.493 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.001, "clf__selection": "random"} |
| ant-1 | ABC | 0.493 | 0.79 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.0001, "clf__selection": "random"} |
| camel-1 | Genetic | 0.490 | 0.29 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1695, "clf__tol": 0.0005132996991867808, "clf__selection": "cyclic"} |
| camel-1 | PSO | 0.490 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.001, "clf__selection": "random"} |
| camel-1 | CDEA | 0.490 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| camel-1 | ABC | 0.490 | 0.79 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| ivy-2 | Genetic | 0.282 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1592, "clf__tol": 0.0009312924857837603, "clf__selection": "cyclic"} |
| ivy-2 | PSO | 0.282 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.0001, "clf__selection": "random"} |
| ivy-2 | CDEA | 0.282 | 0.34 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.001, "clf__selection": "random"} |
| ivy-2 | ABC | 0.282 | 0.8 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.0001, "clf__selection": "random"} |
| jedit-4D | Genetic | 0.629 | 0.2 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1683, "clf__tol": 0.0006866491056557044, "clf__selection": "random"} |
| jedit-4D | PSO | 0.629 | 0.34 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.0001, "clf__selection": "cyclic"} |
| jedit-4D | CDEA | 0.629 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.0001, "clf__selection": "random"} |
| jedit-4D | ABC | 0.629 | 0.75 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.001, "clf__selection": "random"} |
| log4j-1 | Genetic | 0.617 | 0.25 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1901, "clf__tol": 0.0007878387354958853, "clf__selection": "cyclic"} |
| log4j-1 | PSO | 0.617 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| log4j-1 | CDEA | 0.617 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| log4j-1 | ABC | 0.617 | 0.76 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.0001, "clf__selection": "random"} |
| poi-2 | Genetic | 0.222 | 0.34 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1163, "clf__tol": 0.0002901468594269344, "clf__selection": "random"} |
| poi-2 | PSO | 0.222 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.0001, "clf__selection": "cyclic"} |
| poi-2 | CDEA | 0.222 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.0001, "clf__selection": "random"} |
| poi-2 | ABC | 0.222 | 0.8 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.001, "clf__selection": "random"} |
| tomcat | Genetic | 0.290 | 0.28 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1579, "clf__tol": 0.0002598981779152215, "clf__selection": "cyclic"} |
| tomcat | PSO | 0.290 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| tomcat | CDEA | 0.290 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.0001, "clf__selection": "cyclic"} |
| tomcat | ABC | 0.290 | 0.77 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.0001, "clf__selection": "random"} |
| velocity-1 | Genetic | 0.543 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1548, "clf__tol": 0.0006235100271725255, "clf__selection": "cyclic"} |
| velocity-1 | PSO | 0.543 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| velocity-1 | CDEA | 0.543 | 0.34 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.0001, "clf__selection": "random"} |
| velocity-1 | ABC | 0.543 | 0.79 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.0001, "clf__selection": "random"} |
| xalan-2 | Genetic | 0.296 | 0.31 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1633, "clf__tol": 0.000371889318907771, "clf__selection": "cyclic"} |
| xalan-2 | PSO | 0.296 | 0.34 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| xalan-2 | CDEA | 0.296 | 0.34 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| xalan-2 | ABC | 0.296 | 0.78 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.0001, "clf__selection": "random"} |
| xerces-1 | Genetic | 0.505 | 0.27 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1291, "clf__tol": 0.00025548594741575957, "clf__selection": "cyclic"} |
| xerces-1 | PSO | 0.505 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| xerces-1 | CDEA | 0.505 | 0.36 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.0001, "clf__selection": "cyclic"} |

| xerces-1 | ABC | 0.505 | 0.8 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.0001, "clf__selection": "cyclic"} |

## Table 3: Comparison of Different Algorithms and Bio-inspired Approaches

| Dataset | Approach | Algorithm | RMSE | Time (s) | Hyperparameters |
|---------|----------|-----------|------|----------|-----------------|
| ant-1 | Genetic | LinearRegression | 0.379 | 9.5 | {"clf__copy_X": false, "clf__fit_intercept": true, "clf__positive": false} |
| ant-1 | PSO | LinearRegression | 0.379 | 0.36 | {"clf__copy_X": false, "clf__fit_intercept": true, "clf__positive": false} |
| ant-1 | CDEA | LinearRegression | 0.379 | 0.34 | {"clf__copy_X": true, "clf__fit_intercept": true, "clf__positive": false} |
| ant-1 | ABC | LinearRegression | 0.379 | 0.76 | {"clf__copy_X": true, "clf__fit_intercept": true, "clf__positive": false} |
| ant-1 | Genetic | DecisionTreeRegressor | 0.399 | 0.56 | {"clf__max_depth": 4, "clf__min_samples_split": 10, "clf__min_samples_leaf": 1, "clf__random_state": 42} |
| ant-1 | PSO | DecisionTreeRegressor | 0.392 | 0.38 | {"clf__max_depth": 3, "clf__min_samples_split": 5, "clf__min_samples_leaf": 4, "clf__random_state": 42} |
| ant-1 | CDEA | DecisionTreeRegressor | 0.395 | 0.43 | {"clf__max_depth": 3, "clf__min_samples_split": 5, "clf__min_samples_leaf": 2, "clf__random_state": 42} |
| ant-1 | ABC | DecisionTreeRegressor | 0.395 | 0.77 | {"clf__max_depth": 3, "clf__min_samples_split": 5, "clf__min_samples_leaf": 2, "clf__random_state": 42} |
| ant-1 | Genetic | RandomForestRegressor | 0.375 | 4.05 | {"clf__n_estimators": 59, "clf__max_depth": 7, "clf__min_samples_split": 5, "clf__random_state": 42} |
| ant-1 | PSO | RandomForestRegressor | 0.369 | 6.14 | {"clf__n_estimators": 100, "clf__max_depth": 5, "clf__min_samples_split": 2, "clf__random_state": 42} |
| ant-1 | CDEA | RandomForestRegressor | 0.371 | 5.87 | {"clf__n_estimators": 100, "clf__max_depth": 5, "clf__min_samples_split": 5, "clf__random_state": 42} |
| ant-1 | ABC | RandomForestRegressor | 0.371 | 11.29 | {"clf__n_estimators": 100, "clf__max_depth": 5, "clf__min_samples_split": 5, "clf__random_state": 42} |
| ant-1 | Genetic | GradientBoostingRegressor | 0.379 | 2.76 | {"clf__n_estimators": 69, "clf__learning_rate": 0.059574873458136744, "clf__max_depth": 3, "clf__subsample": 0.8518340713515969} |
| ant-1 | PSO | GradientBoostingRegressor | 0.379 | 2.37 | {"clf__n_estimators": 50, "clf__learning_rate": 0.05, "clf__max_depth": 3, "clf__subsample": 0.7} |
| ant-1 | CDEA | GradientBoostingRegressor | 0.379 | 3.62 | {"clf__n_estimators": 50, "clf__learning_rate": 0.05, "clf__max_depth": 3, "clf__subsample": 0.7} |
| ant-1 | ABC | GradientBoostingRegressor | 0.379 | 6.85 | {"clf__n_estimators": 50, "clf__learning_rate": 0.05, "clf__max_depth": 3, "clf__subsample": 0.7} |
| ant-1 | Genetic | Lasso | 0.493 | 0.33 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1615, "clf__tol": 0.00033822480140215046, "clf__selection": "cyclic"} |
| ant-1 | PSO | Lasso | 0.493 | 0.34 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 1000, "clf__tol": 0.001, "clf__selection": "cyclic"} |
| ant-1 | CDEA | Lasso | 0.493 | 0.35 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.001, "clf__selection": "random"} |
| ant-1 | ABC | Lasso | 0.493 | 0.79 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__max_iter": 2000, "clf__tol": 0.0001, "clf__selection": "random"} |
| ant-1 | Genetic | Ridge | 0.379 | 0.22 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__tol": 0.0004812601007838968, "clf__solver": "auto"} |
| ant-1 | PSO | Ridge | 0.379 | 0.36 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__tol": 0.0001, "clf__solver": "auto"} |
| ant-1 | CDEA | Ridge | 0.379 | 0.36 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__tol": 0.001, "clf__solver": "auto"} |
| ant-1 | ABC | Ridge | 0.379 | 0.78 | {"clf__alpha": 1.0, "clf__fit_intercept": true, "clf__tol": 0.0001, "clf__solver": "auto"} |
| ant-1 | Genetic | ElasticNet | 0.381 | 0.4 | {"clf__alpha": 0.176971362875233, "clf__l1_ratio": 0.1324163961684037, "clf__fit_intercept": true} |
| ant-1 | PSO | ElasticNet | 0.378 | 0.35 | {"clf__alpha": 0.1, "clf__l1_ratio": 0.1, "clf__fit_intercept": true} |
| ant-1 | CDEA | ElasticNet | 0.379 | 0.35 | {"clf__alpha": 0.001, "clf__l1_ratio": 0.1, "clf__fit_intercept": true} |
| ant-1 | ABC | ElasticNet | 0.378 | 0.74 | {"clf__alpha": 0.1, "clf__l1_ratio": 0.1, "clf__fit_intercept": true} |
| ant-1 | Genetic | KNeighborsRegressor | 0.421 | 0.7 | {"clf__n_neighbors": 7, "clf__weights": "distance", "clf__algorithm": "ball_tree"} |
| ant-1 | PSO | KNeighborsRegressor | 0.421 | 0.37 | {"clf__n_neighbors": 7, "clf__weights": "distance", "clf__algorithm": "ball_tree"} |
| ant-1 | CDEA | KNeighborsRegressor | 0.421 | 0.35 | {"clf__n_neighbors": 7, "clf__weights": "distance", "clf__algorithm": "ball_tree"} |
| ant-1 | ABC | KNeighborsRegressor | 0.421 | 0.78 | {"clf__n_neighbors": 7, "clf__weights": "distance", "clf__algorithm": "kd_tree"} |
| ant-1 | Genetic | SVR | 0.397 | 2.02 | {"clf__kernel": "linear", "clf__C": 2.108039539023126, "clf__epsilon": 0.09813636902574707, "clf__gamma": "scale"} |
| ant-1 | PSO | SVR | 0.384 | 3.07 | {"clf__kernel": "linear", "clf__C": 0.1, "clf__epsilon": 0.2, "clf__gamma": "scale"} |
| ant-1 | CDEA | SVR | 0.384 | 7.09 | {"clf__kernel": "linear", "clf__C": 0.1, "clf__epsilon": 0.2, "clf__gamma": "auto"} |
| ant-1 | ABC | SVR | 0.384 | 6.9 | {"clf__kernel": "linear", "clf__C": 0.1, "clf__epsilon": 0.2, "clf__gamma": "scale"} |
| ant-1 | Genetic | MLPRegressor | 0.455 | 4.2 | {"clf__hidden_layer_sizes": [100], "clf__activation": "tanh", "clf__alpha": 0.00994291441327896, "clf__learning_rate_init": 0.0028108271528846335} |
| ant-1 | PSO | MLPRegressor | 0.419 | 6.91 | {"clf__hidden_layer_sizes": [100], "clf__activation": "tanh", "clf__alpha": 0.001, "clf__learning_rate_init": 0.001} |
| ant-1 | CDEA | MLPRegressor | 0.419 | 6.02 | {"clf__hidden_layer_sizes": [100], "clf__activation": "tanh", "clf__alpha": 0.001, "clf__learning_rate_init": 0.001} |
| ant-1 | ABC | MLPRegressor | 0.419 | 13.88 | {"clf__hidden_layer_sizes": [100], "clf__activation": "tanh", "clf__alpha": 0.001, "clf__learning_rate_init": 0.001} |
| ant-1 | Genetic | XGBRegressor | 0.383 | 2.83 | {"clf__learning_rate": 0.029902757668529505, "clf__n_estimators": 96, "clf__max_depth": 4, "clf__subsample": 0.8350111799995532, "clf__colsample_bytree": 0.8790946006929988} |
| ant-1 | PSO | XGBRegressor | 0.403 | 2.34 | {"clf__learning_rate": 0.01, "clf__n_estimators": 100, "clf__max_depth": 3, "clf__subsample": 0.8, "clf__colsample_bytree": 0.8} |
| ant-1 | CDEA | XGBRegressor | 0.376 | 1.33 | {"clf__learning_rate": 0.1, "clf__n_estimators": 50, "clf__max_depth": 3, "clf__subsample": 0.8, "clf__colsample_bytree": 0.8} |
| ant-1 | ABC | XGBRegressor | 0.389 | 3.61 | {"clf__learning_rate": 0.1, "clf__n_estimators": 50, "clf__max_depth": 3, "clf__subsample": 0.8, "clf__colsample_bytree": 1.0} |
| camel-1 | Genetic | LinearRegression | 0.460 | 0.11 | {"clf__copy_X": false, "clf__fit_intercept": true, "clf__positive": false} |
| camel-1 | PSO | LinearRegression | 0.460 | 0.34 | {"clf__copy_X": false, "clf__fit_intercept": true, "clf__positive": false} |
| camel-1 | CDEA | LinearRegression | 0.460 | 0.35 | {"clf__copy_X": false, "clf__fit_intercept": true, "clf__positive": false} |
| camel-1 | ABC | LinearRegression | 0.460 | 0.75 | {"clf__copy_X": false, "clf__fit_intercept": true, "clf__positive": false} |
| camel-1 | Genetic | DecisionTreeRegressor | 0.469 | 0.21 | {"clf__max_depth": 3, "clf__min_samples_split": 7, "clf__min_samples_leaf": 4, "clf__random_state": 42} |
| camel-1 | PSO | DecisionTreeRegressor | 0.469 | 0.35 | {"clf__max_depth": 3, "clf__min_samples_split": 2, "clf__min_samples_leaf": 4, "clf__random_state": 42} |
| camel-1 | CDEA | DecisionTreeRegressor | 0.469 | 0.35 | {"clf__max_depth": 3, "clf__min_samples_split": 5, "clf__min_samples_leaf": 4, "clf__random_state": 42} |
| camel-1 | ABC | DecisionTreeRegressor | 0.469 | 0.72 | {"clf__max_depth": 3, "clf__min_samples_split": 5, "clf__min_samples_leaf": 4, "clf__random_state": 42} |
| camel-1 | Genetic | RandomForestRegressor | 0.447 | 2.65 | {"clf__n_estimators": 93, "clf__max_depth": 8, "clf__min_samples_split": 5, "clf__random_state": 42} |
... (table continues, same as generated_tables_utf8.md) ...

| Dataset | Algorithm | Genetic Time (s) | PSO Time (s) | CDEA Time (s) | ABC Time (s) |
|---------|-----------|------------------:|--------------:|-------------:|------------:|
| ant-1 | LinearRegression | 9.5 | 0.36 | 0.34 | 0.76 |
| ant-1 | DecisionTreeRegressor | 0.56 | 0.38 | 0.43 | 0.77 |
| ant-1 | RandomForestRegressor | 4.05 | 6.14 | 5.87 | 11.29 |
| ant-1 | GradientBoostingRegressor | 2.76 | 2.37 | 3.62 | 6.85 |
| ant-1 | Lasso | 0.33 | 0.34 | 0.35 | 0.79 |
| ant-1 | Ridge | 0.22 | 0.36 | 0.36 | 0.78 |
| ant-1 | ElasticNet | 0.4 | 0.35 | 0.35 | 0.74 |
| ant-1 | KNeighborsRegressor | 0.7 | 0.37 | 0.35 | 0.78 |
| ant-1 | SVR | 2.02 | 3.07 | 7.09 | 6.9 |
| ant-1 | MLPRegressor | 4.2 | 6.91 | 6.02 | 13.88 |
| ant-1 | XGBRegressor | 2.83 | 2.34 | 1.33 | 3.61 |


