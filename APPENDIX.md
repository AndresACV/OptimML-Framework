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

## Table 2: Comparison of Genetic and Exhaustive approaches with Lasso algorithm

| Dataset | Approach | Algorithm | RMSE | Time (s) | alpha | fit_intercept | max_iter | tol | selection |
|---------|----------|-----------|------|----------|-------|---------------|----------|-----|-----------|
| poi-2 | Exhaustive | Lasso | 0.222 | 2.486 | 1 | True | 1405 | 3.66e-4 | cyclic |
| poi-2 | Genetic | Lasso | 0.222 | 0.052 | 1 | True | 1000 | 1.00e-4 | cyclic |

## Table 3: Comparison of Different Algorithms and Approaches

| Dataset | Approach | Algorithm | RMSE | Time (s) | clf__alpha | clf__fit_intercept | clf__max_iter | clf__tol | clf__selection | clf__colsample_bytree | clf__learning_rate | clf__max_depth | clf__n_estimators | clf__min_samples_split |
|---------|----------|-----------|------|----------|------------|-------------------|---------------|----------|----------------|----------------------|-------------------|---------------|------------------|----------------------|
| poi-2 | Exhaustive | Lasso | 0.222 | 2.486 | 1 | True | 1405 | 3.66e-4 | cyclic | -- | -- | -- | -- | -- |
| poi-2 | Genetic | XGBRegressor | 0.222 | 0.316 | -- | -- | -- | -- | -- | 1 | 0.01 | 3 | 100 | -- |
| tomcat | Exhaustive | RandomForestRegressor | 0.230 | 7.271 | -- | -- | -- | -- | -- | -- | -- | 6 | 89 | 4 |

## Table 4: Training time comparison between Genetic and Exhaustive approaches with performance statistics

| Dataset | Algorithm | Exhaustive Time (s) | Genetic Time (s) | Speedup Factor | Time Overhead (%) |
|---------|-----------|-------------------|-----------------|----------------|------------------|
| ant-1 | LinearRegression | 6.52 | 0.05 | 130.40x | 13,040 |
| ant-1 | DecisionTreeRegressor | 3.23 | 0.12 | 26.92x | 2,592 |
| ant-1 | RandomForestRegressor | 8.38 | 0.96 | 8.73x | 773 |
| ant-1 | Lasso | 3.28 | 0.05 | 65.60x | 6,460 |
| ant-1 | Ridge | 3.09 | 0.05 | 61.80x | 6,080 |
| ant-1 | KNeighborsRegressor | 3.57 | 0.07 | 51.00x | 5,000 |
| ant-1 | XGBRegressor | 4.21 | 0.78 | 5.40x | 440 |
| camel-1 | LinearRegression | 2.76 | 0.04 | 69.00x | 6,800 |
| camel-1 | DecisionTreeRegressor | 2.59 | 0.10 | 25.90x | 2,490 |
| camel-1 | RandomForestRegressor | 5.79 | 0.54 | 10.72x | 972 |
| camel-1 | Lasso | 2.56 | 0.05 | 51.20x | 5,020 |
| camel-1 | Ridge | 2.87 | 0.04 | 71.75x | 7,075 |
| camel-1 | KNeighborsRegressor | 2.62 | 0.06 | 43.67x | 4,267 |
| camel-1 | XGBRegressor | 3.54 | 0.38 | 9.32x | 832 |
| ivy-2 | LinearRegression | 2.26 | 0.05 | 45.20x | 4,420 |
| ivy-2 | DecisionTreeRegressor | 2.75 | 0.09 | 30.56x | 2,956 |
| ivy-2 | RandomForestRegressor | 5.53 | 0.46 | 12.02x | 1,102 |
| ivy-2 | Lasso | 2.43 | 0.05 | 48.60x | 4,760 |
| ivy-2 | Ridge | 2.81 | 0.04 | 70.25x | 6,925 |
| ivy-2 | KNeighborsRegressor | 2.56 | 0.06 | 42.67x | 4,167 |
| ivy-2 | XGBRegressor | 3.47 | 0.35 | 9.91x | 891 |
| jedit-4D | LinearRegression | 2.67 | 0.04 | 66.75x | 6,575 |
| jedit-4D | DecisionTreeRegressor | 2.44 | 0.09 | 27.11x | 2,611 |
| jedit-4D | RandomForestRegressor | 5.76 | 0.47 | 12.26x | 1,126 |
| jedit-4D | Lasso | 2.35 | 0.05 | 47.00x | 4,600 |
| jedit-4D | Ridge | 2.64 | 0.04 | 66.00x | 6,500 |
| jedit-4D | KNeighborsRegressor | 2.64 | 0.06 | 44.00x | 4,300 |
| jedit-4D | XGBRegressor | 3.23 | 0.35 | 9.23x | 823 |
| log4j-1 | LinearRegression | 2.59 | 0.04 | 64.75x | 6,375 |
| log4j-1 | DecisionTreeRegressor | 2.80 | 0.09 | 31.11x | 3,011 |
| log4j-1 | RandomForestRegressor | 5.31 | 0.46 | 11.54x | 1,054 |
| log4j-1 | Lasso | 2.92 | 0.05 | 58.40x | 5,740 |
| log4j-1 | Ridge | 2.83 | 0.04 | 70.75x | 6,975 |
| log4j-1 | KNeighborsRegressor | 2.80 | 0.06 | 46.67x | 4,567 |
| log4j-1 | XGBRegressor | 3.31 | 0.36 | 9.19x | 819 |
| poi-2 | LinearRegression | 2.60 | 0.04 | 65.00x | 6,400 |
| poi-2 | DecisionTreeRegressor | 2.71 | 0.09 | 30.11x | 2,911 |
| poi-2 | RandomForestRegressor | 5.49 | 0.40 | 13.73x | 1,273 |
| poi-2 | Lasso | 2.49 | 0.05 | 49.80x | 4,880 |
| poi-2 | Ridge | 2.78 | 0.04 | 69.50x | 6,850 |
| poi-2 | KNeighborsRegressor | 2.42 | 0.06 | 40.33x | 3,933 |
| poi-2 | XGBRegressor | 3.29 | 0.32 | 10.28x | 928 |
| tomcat | LinearRegression | 2.62 | 0.05 | 52.40x | 5,140 |
| tomcat | DecisionTreeRegressor | 2.47 | 0.10 | 24.70x | 2,370 |
| tomcat | RandomForestRegressor | 7.27 | 0.61 | 11.92x | 1,092 |
| tomcat | Lasso | 2.80 | 0.05 | 56.00x | 5,500 |
| tomcat | Ridge | 2.60 | 0.04 | 65.00x | 6,400 |
| tomcat | KNeighborsRegressor | 2.75 | 0.06 | 45.83x | 4,483 |
| tomcat | XGBRegressor | 4.18 | 0.55 | 7.60x | 660 |
| velocity-1 | LinearRegression | 2.89 | 0.04 | 72.25x | 7,125 |
| velocity-1 | DecisionTreeRegressor | 2.80 | 0.09 | 31.11x | 3,011 |
| velocity-1 | RandomForestRegressor | 6.04 | 0.43 | 14.05x | 1,305 |
| velocity-1 | Lasso | 2.41 | 0.05 | 48.20x | 4,720 |
| velocity-1 | Ridge | 2.71 | 0.04 | 67.75x | 6,675 |
| velocity-1 | KNeighborsRegressor | 2.54 | 0.05 | 50.80x | 4,980 |
| velocity-1 | XGBRegressor | 3.53 | 0.35 | 10.09x | 909 |
| xalan-2 | LinearRegression | 2.85 | 0.05 | 57.00x | 5,600 |
| xalan-2 | DecisionTreeRegressor | 2.88 | 0.09 | 32.00x | 3,100 |
| xalan-2 | RandomForestRegressor | 6.94 | 0.71 | 9.77x | 877 |
| xalan-2 | Lasso | 2.74 | 0.05 | 54.80x | 5,380 |
| xalan-2 | Ridge | 2.64 | 0.04 | 66.00x | 6,500 |
| xalan-2 | KNeighborsRegressor | 2.79 | 0.06 | 46.50x | 4,550 |
| xalan-2 | XGBRegressor | 3.62 | 0.60 | 6.03x | 503 |
| xerces-1 | LinearRegression | 2.91 | 0.04 | 72.75x | 7,175 |
| xerces-1 | DecisionTreeRegressor | 2.75 | 0.09 | 30.56x | 2,956 |
| xerces-1 | RandomForestRegressor | 7.36 | 0.53 | 13.89x | 1,289 |
| xerces-1 | Lasso | 2.85 | 0.05 | 57.00x | 5,600 |
| xerces-1 | Ridge | 2.45 | 0.04 | 61.25x | 6,025 |
| xerces-1 | KNeighborsRegressor | 2.89 | 0.06 | 48.17x | 4,717 |
| xerces-1 | XGBRegressor | 3.58 | 0.42 | 8.52x | 752 |

## Table 5: RMSE comparison between Genetic and Exhaustive approaches to evaluate performance similarity

| Dataset | Algorithm | RMSE (Genetic) | RMSE (Exhaustive) | Absolute Difference | Relative Difference (%) |
|---------|-----------|----------------|-------------------|-------------------|----------------------|
| ant-1 | LinearRegression | 0.376 | 0.379 | 0.003 | 0.79 |
| ant-1 | DecisionTreeRegressor | 0.392 | 0.395 | 0.003 | 0.68 |
| ant-1 | RandomForestRegressor | 0.371 | 0.371 | 0.001 | 0.24 |
| ant-1 | Lasso | 0.493 | 0.493 | 0.000 | 0.00 |
| ant-1 | Ridge | 0.379 | 0.379 | 0.000 | 0.00 |
| ant-1 | KNeighborsRegressor | 0.421 | 0.421 | 0.000 | 0.00 |
| ant-1 | XGBRegressor | 0.384 | 0.389 | 0.004 | 1.15 |
| camel-1 | LinearRegression | 0.460 | 0.460 | 0.000 | 0.00 |
| camel-1 | DecisionTreeRegressor | 0.469 | 0.469 | 0.000 | 0.00 |
| camel-1 | RandomForestRegressor | 0.448 | 0.447 | 0.001 | 0.27 |
| camel-1 | Lasso | 0.490 | 0.490 | 0.000 | 0.00 |
| camel-1 | Ridge | 0.460 | 0.460 | 0.000 | 0.00 |
| camel-1 | KNeighborsRegressor | 0.461 | 0.461 | 0.000 | 0.00 |
| camel-1 | XGBRegressor | 0.450 | 0.454 | 0.004 | 0.98 |
| ivy-2 | LinearRegression | 0.253 | 0.253 | 0.000 | 0.00 |
| ivy-2 | DecisionTreeRegressor | 0.268 | 0.257 | 0.012 | 4.55 |
| ivy-2 | RandomForestRegressor | 0.266 | 0.260 | 0.006 | 2.19 |
| ivy-2 | Lasso | 0.282 | 0.282 | 0.000 | 0.00 |
| ivy-2 | Ridge | 0.253 | 0.253 | 0.000 | 0.00 |
| ivy-2 | KNeighborsRegressor | 0.277 | 0.277 | 0.000 | 0.00 |
| ivy-2 | XGBRegressor | 0.260 | 0.268 | 0.008 | 2.96 |
| jedit-4D | LinearRegression | 0.401 | 0.401 | 0.000 | 0.00 |
| jedit-4D | DecisionTreeRegressor | 0.500 | 0.500 | 0.000 | 0.00 |
| jedit-4D | RandomForestRegressor | 0.480 | 0.487 | 0.007 | 1.53 |
| jedit-4D | Lasso | 0.629 | 0.629 | 0.000 | 0.00 |
| jedit-4D | Ridge | 0.401 | 0.401 | 0.000 | 0.00 |
| jedit-4D | KNeighborsRegressor | 0.555 | 0.562 | 0.007 | 1.25 |
| jedit-4D | XGBRegressor | 0.511 | 0.536 | 0.025 | 4.91 |
| log4j-1 | LinearRegression | 0.393 | 0.387 | 0.005 | 1.36 |
| log4j-1 | DecisionTreeRegressor | 0.518 | 0.559 | 0.041 | 7.94 |
| log4j-1 | RandomForestRegressor | 0.504 | 0.501 | 0.003 | 0.61 |
| log4j-1 | Lasso | 0.617 | 0.617 | 0.000 | 0.00 |
| log4j-1 | Ridge | 0.387 | 0.387 | 0.000 | 0.00 |
| log4j-1 | KNeighborsRegressor | 0.515 | 0.482 | 0.033 | 6.92 |
| log4j-1 | XGBRegressor | 0.509 | 0.527 | 0.017 | 3.39 |
| poi-2 | LinearRegression | 0.265 | 0.265 | 0.000 | 0.00 |
| poi-2 | DecisionTreeRegressor | 0.233 | 0.233 | 0.000 | 0.00 |
| poi-2 | RandomForestRegressor | 0.274 | 0.258 | 0.016 | 6.19 |
| poi-2 | Lasso | 0.222 | 0.222 | 0.000 | 0.00 |
| poi-2 | Ridge | 0.265 | 0.265 | 0.000 | 0.00 |
| poi-2 | KNeighborsRegressor | 0.233 | 0.233 | 0.000 | 0.00 |
| poi-2 | XGBRegressor | 0.229 | 0.222 | 0.007 | 3.01 |
| tomcat | LinearRegression | 0.236 | 0.236 | 0.000 | 0.00 |
| tomcat | DecisionTreeRegressor | 0.248 | 0.252 | 0.004 | 1.64 |
| tomcat | RandomForestRegressor | 0.230 | 0.236 | 0.006 | 2.78 |
| tomcat | Lasso | 0.290 | 0.290 | 0.000 | 0.00 |
| tomcat | Ridge | 0.236 | 0.236 | 0.000 | 0.00 |
| tomcat | KNeighborsRegressor | 0.262 | 0.265 | 0.003 | 1.03 |
| tomcat | XGBRegressor | 0.238 | 0.239 | 0.002 | 0.75 |
| velocity-1 | LinearRegression | 0.456 | 0.456 | 0.000 | 0.00 |
| velocity-1 | DecisionTreeRegressor | 0.635 | 0.682 | 0.047 | 7.40 |
| velocity-1 | RandomForestRegressor | 0.548 | 0.544 | 0.004 | 0.65 |
| velocity-1 | Lasso | 0.543 | 0.543 | 0.000 | 0.00 |
| velocity-1 | Ridge | 0.456 | 0.456 | 0.000 | 0.00 |
| velocity-1 | KNeighborsRegressor | 0.538 | 0.538 | 0.000 | 0.00 |
| velocity-1 | XGBRegressor | 0.511 | 0.503 | 0.008 | 1.57 |
| xalan-2 | LinearRegression | 0.268 | 0.268 | 0.000 | 0.00 |
| xalan-2 | DecisionTreeRegressor | 0.309 | 0.309 | 0.000 | 0.00 |
| xalan-2 | RandomForestRegressor | 0.272 | 0.270 | 0.002 | 0.88 |
| xalan-2 | Lasso | 0.296 | 0.296 | 0.000 | 0.00 |
| xalan-2 | Ridge | 0.268 | 0.268 | 0.000 | 0.00 |
| xalan-2 | KNeighborsRegressor | 0.273 | 0.273 | 0.000 | 0.00 |
| xalan-2 | XGBRegressor | 0.266 | 0.272 | 0.006 | 2.15 |
| xerces-1 | LinearRegression | 0.408 | 0.408 | 0.000 | 0.00 |
| xerces-1 | DecisionTreeRegressor | 0.433 | 0.492 | 0.058 | 13.39 |
| xerces-1 | RandomForestRegressor | 0.404 | 0.402 | 0.002 | 0.45 |
| xerces-1 | Lasso | 0.505 | 0.505 | 0.000 | 0.00 |
| xerces-1 | Ridge | 0.408 | 0.408 | 0.000 | 0.00 |
| xerces-1 | KNeighborsRegressor | 0.419 | 0.429 | 0.010 | 2.45 |
| xerces-1 | XGBRegressor | 0.419 | 0.398 | 0.021 | 5.11 |
