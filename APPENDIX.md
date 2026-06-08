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

## Table 2: Comparison of Lasso across bioinspired approaches

| Dataset | Approach | Algorithm | RMSE | Time (s) | alpha | fit_intercept | max_iter | tol | selection |
|---------|----------|-----------|------|----------|-------|---------------|----------|-----|-----------|
| poi-2 | Genetic | Lasso | 0.2218620362949892 | 0.3353297710418701 | 1 | True | 1163 | 0.0002901468594269344 | random |
| poi-2 | PSO | Lasso | 0.2218620362949892 | 0.3476696014404297 | 1 | True | 1000 | 0.0001 | cyclic |
| poi-2 | CDEA | Lasso | 0.2218620362949892 | 0.3487777709960938 | 1 | True | 2000 | 0.0001 | random |
| poi-2 | ABC | Lasso | 0.2218620362949892 | 0.8038005828857422 | 1 | True | 2000 | 0.001 | random |

## Table 3: Comparison of Different Algorithms and Bioinspired Approaches

| Dataset | Approach | Algorithm | RMSE | Time (s) | clf__alpha | clf__fit_intercept | clf__max_iter | clf__tol | clf__selection | clf__colsample_bytree | clf__learning_rate | clf__max_depth | clf__n_estimators | clf__min_samples_split |
|---------|----------|-----------|------|----------|------------|-------------------|---------------|----------|----------------|----------------------|-------------------|---------------|------------------|----------------------|
| poi-2 | Genetic | Lasso | 0.2218620362949892 | 0.34 seconds | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| poi-2 | PSO | Lasso | 0.2218620362949892 | 0.35 seconds | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| poi-2 | CDEA | Lasso | 0.2218620362949892 | 0.35 seconds | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| poi-2 | ABC | Lasso | 0.2218620362949892 | 0.80 seconds | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| poi-2 | Genetic | XGBRegressor | 0.2368127468642922 | 0.75 seconds | -- | -- | -- | -- | -- | 0.8705562941036851 | 0.03541381321878839 | 3 | 76 | -- |
| poi-2 | PSO | XGBRegressor | 0.2235375323017195 | 0.88 seconds | -- | -- | -- | -- | -- | 0.8 | 0.01 | 3 | 100 | -- |
| poi-2 | CDEA | XGBRegressor | 0.2222223430866861 | 0.78 seconds | -- | -- | -- | -- | -- | 1.0 | 0.01 | 3 | 100 | -- |
| poi-2 | ABC | XGBRegressor | 0.2501156367404922 | 1.68 seconds | -- | -- | -- | -- | -- | 1.0 | 0.1 | 3 | 50 | -- |
| tomcat | Genetic | RandomForestRegressor | 0.2363814958163347 | 2.96 seconds | -- | -- | -- | -- | -- | -- | -- | 5 | 52 | 2 |
| tomcat | PSO | RandomForestRegressor | 0.2360555139550304 | 3.78 seconds | -- | -- | -- | -- | -- | -- | -- | 5 | 50 | 2 |
| tomcat | CDEA | RandomForestRegressor | 0.2360555139550304 | 3.94 seconds | -- | -- | -- | -- | -- | -- | -- | 5 | 50 | 2 |
| tomcat | ABC | RandomForestRegressor | 0.2360555139550304 | 8.16 seconds | -- | -- | -- | -- | -- | -- | -- | 5 | 50 | 2 |

## Table 4: Training time comparison among Bioinspired approaches

| Dataset | Algorithm | Genetic Time (s) | PSO Time (s) | CDEA Time (s) | ABC Time (s) | PSO/Genetic Speedup | CDEA/Genetic Speedup | ABC/Genetic Speedup |
|---------|-----------|------------------|--------------|---------------|--------------|--------------------|--------------------|--------------------|
| ant-1 | LinearRegression | 9.5 | 0.36 | 0.34 | 0.76 | 26.39 | 27.94 | 12.5 |
| ant-1 | DecisionTreeRegressor | 0.56 | 0.38 | 0.43 | 0.77 | 1.47 | 1.3 | 0.73 |
| ant-1 | RandomForestRegressor | 4.05 | 6.14 | 5.87 | 11.29 | 0.66 | 0.69 | 0.36 |
| ant-1 | Lasso | 0.33 | 0.34 | 0.35 | 0.79 | 0.97 | 0.94 | 0.42 |
| ant-1 | XGBRegressor | 2.83 | 2.34 | 1.33 | 3.61 | 1.21 | 2.13 | 0.78 |
| camel-1 | LinearRegression | 0.11 | 0.34 | 0.35 | 0.75 | 0.32 | 0.31 | 0.15 |
| camel-1 | DecisionTreeRegressor | 0.21 | 0.35 | 0.35 | 0.72 | 0.6 | 0.6 | 0.29 |
| camel-1 | RandomForestRegressor | 2.65 | 3.69 | 3.33 | 8.46 | 0.72 | 0.8 | 0.31 |
| camel-1 | Lasso | 0.29 | 0.35 | 0.35 | 0.79 | 0.83 | 0.83 | 0.37 |
| camel-1 | XGBRegressor | 1.31 | 0.73 | 1.01 | 2.01 | 1.79 | 1.3 | 0.65 |
| ivy-2 | LinearRegression | 0.11 | 0.39 | 0.35 | 0.77 | 0.28 | 0.31 | 0.14 |
| ivy-2 | DecisionTreeRegressor | 0.31 | 0.35 | 0.36 | 0.75 | 0.89 | 0.86 | 0.41 |
| ivy-2 | RandomForestRegressor | 3.06 | 3.48 | 3.05 | 7.62 | 0.88 | 1.0 | 0.4 |
| ivy-2 | Lasso | 0.35 | 0.35 | 0.34 | 0.8 | 1.0 | 1.03 | 0.44 |
| ivy-2 | XGBRegressor | 0.87 | 0.92 | 0.83 | 1.83 | 0.95 | 1.05 | 0.48 |
| jedit-4D | LinearRegression | 0.1 | 0.35 | 0.34 | 0.76 | 0.29 | 0.29 | 0.13 |
| jedit-4D | DecisionTreeRegressor | 0.26 | 0.35 | 0.35 | 0.74 | 0.74 | 0.74 | 0.35 |
| jedit-4D | RandomForestRegressor | 2.41 | 3.4 | 4.03 | 7.29 | 0.71 | 0.6 | 0.33 |
| jedit-4D | Lasso | 0.2 | 0.34 | 0.35 | 0.75 | 0.59 | 0.57 | 0.27 |
| jedit-4D | XGBRegressor | 1.05 | 1.11 | 0.89 | 1.84 | 0.95 | 1.18 | 0.57 |
| log4j-1 | LinearRegression | 0.13 | 0.35 | 0.34 | 0.76 | 0.37 | 0.38 | 0.17 |
| log4j-1 | DecisionTreeRegressor | 0.25 | 0.35 | 0.35 | 0.74 | 0.71 | 0.71 | 0.34 |
| log4j-1 | RandomForestRegressor | 1.63 | 3.41 | 3.64 | 7.47 | 0.48 | 0.45 | 0.22 |
| log4j-1 | Lasso | 0.25 | 0.35 | 0.35 | 0.76 | 0.71 | 0.71 | 0.33 |
| log4j-1 | XGBRegressor | 0.86 | 0.95 | 0.92 | 1.9 | 0.91 | 0.93 | 0.45 |
| poi-2 | LinearRegression | 0.08 | 0.34 | 0.34 | 0.78 | 0.24 | 0.24 | 0.1 |
| poi-2 | DecisionTreeRegressor | 0.27 | 0.36 | 0.35 | 0.78 | 0.75 | 0.77 | 0.35 |
| poi-2 | RandomForestRegressor | 1.77 | 3.76 | 2.54 | 7.02 | 0.47 | 0.7 | 0.25 |
| poi-2 | Lasso | 0.34 | 0.35 | 0.35 | 0.8 | 0.97 | 0.97 | 0.42 |
| poi-2 | XGBRegressor | 0.75 | 0.88 | 0.78 | 1.68 | 0.85 | 0.96 | 0.45 |
| tomcat | LinearRegression | 0.14 | 0.36 | 0.34 | 0.75 | 0.39 | 0.41 | 0.19 |
| tomcat | DecisionTreeRegressor | 0.28 | 0.35 | 0.35 | 0.77 | 0.8 | 0.8 | 0.36 |
| tomcat | RandomForestRegressor | 2.96 | 3.78 | 3.94 | 8.16 | 0.78 | 0.75 | 0.36 |
| tomcat | Lasso | 0.28 | 0.35 | 0.35 | 0.77 | 0.8 | 0.8 | 0.36 |
| tomcat | XGBRegressor | 1.39 | 1.04 | 1.51 | 2.8 | 1.34 | 0.92 | 0.5 |
| velocity-1 | LinearRegression | 0.11 | 0.35 | 0.34 | 0.75 | 0.31 | 0.32 | 0.15 |
| velocity-1 | DecisionTreeRegressor | 0.36 | 0.35 | 0.35 | 0.75 | 1.03 | 1.03 | 0.48 |
| velocity-1 | RandomForestRegressor | 2.47 | 3.25 | 2.88 | 6.37 | 0.76 | 0.86 | 0.39 |
| velocity-1 | Lasso | 0.35 | 0.35 | 0.34 | 0.79 | 1.0 | 1.03 | 0.44 |
| velocity-1 | XGBRegressor | 0.93 | 0.91 | 0.84 | 1.81 | 1.02 | 1.11 | 0.51 |
| xalan-2 | LinearRegression | 0.13 | 0.34 | 0.34 | 0.76 | 0.38 | 0.38 | 0.17 |
| xalan-2 | DecisionTreeRegressor | 0.34 | 0.35 | 0.35 | 0.78 | 0.97 | 0.97 | 0.44 |
| xalan-2 | RandomForestRegressor | 4.34 | 4.81 | 4.74 | 11.3 | 0.9 | 0.92 | 0.38 |
| xalan-2 | Lasso | 0.31 | 0.34 | 0.34 | 0.78 | 0.91 | 0.91 | 0.4 |
| xalan-2 | XGBRegressor | 1.58 | 1.49 | 1.4 | 2.52 | 1.06 | 1.13 | 0.63 |
| xerces-1 | LinearRegression | 0.15 | 0.35 | 0.35 | 0.79 | 0.43 | 0.43 | 0.19 |
| xerces-1 | DecisionTreeRegressor | 0.31 | 0.37 | 0.35 | 0.78 | 0.84 | 0.89 | 0.4 |
| xerces-1 | RandomForestRegressor | 2.57 | 2.76 | 4.46 | 9.15 | 0.93 | 0.58 | 0.28 |
| xerces-1 | Lasso | 0.27 | 0.35 | 0.36 | 0.8 | 0.77 | 0.75 | 0.34 |
| xerces-1 | XGBRegressor | 0.87 | 0.93 | 0.97 | 2.27 | 0.94 | 0.9 | 0.38 |

## Table 5: RMSE comparison between Bioinspired approaches

| Dataset | Algorithm | RMSE (Genetic) | RMSE (PSO) | RMSE (CDEA) | RMSE (ABC) | Diff PSO | Diff CDEA | Diff ABC |
|---------|-----------|----------------|------------|------------|-----------|----------|----------|---------|
| ant-1 | LinearRegression | 0.3793709331879233 | 0.3793709331879233 | 0.3793709331879233 | 0.3793709331879233 | 0.0 | 0.0 | 0.0 |
| ant-1 | DecisionTreeRegressor | 0.3987188591868169 | 0.3923454296145987 | 0.3950080377234149 | 0.3950080377234149 | 0.006373 | 0.003711 | 0.003711 |
| ant-1 | RandomForestRegressor | 0.3750473806471075 | 0.3692668466646321 | 0.3705267192161611 | 0.3705267192161611 | 0.005781 | 0.004521 | 0.004521 |
| ant-1 | GradientBoostingRegressor | 0.3792011237614747 | 0.3792743998748819 | 0.3792743998748819 | 0.3792743998748819 | 7.3e-05 | 7.3e-05 | 7.3e-05 |
| ant-1 | Lasso | 0.4932075963235883 | 0.4932075963235883 | 0.4932075963235883 | 0.4932075963235883 | 0.0 | 0.0 | 0.0 |
| ant-1 | Ridge | 0.3792726376497357 | 0.3792726376497357 | 0.3792726376497357 | 0.3792726376497357 | 0.0 | 0.0 | 0.0 |
