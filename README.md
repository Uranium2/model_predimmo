# Model Predimmo

Model Predimmo repository's job is to **train a deep learning model and made prediction** on real estate adverts from Leboncoin available in our AWS RDS with ```main.py```

* ```main.py``` script call ```preprocessing_1year.py``` &  ```preprocessing_3years.py``` scripts whom **pre-processing** our datasets from datagouv.fr.
* That allow ```model_1year.py``` &  ```model_3years.py``` scripts to **train our models** whom are **saved** in our daily backup model in ```./save/models/``` folder.
* These **artifact models** are used in our ```predict_1year.py``` & ```predict_3years.py``` scripts to made **predictions** on real estate adverts from Leboncoin.
* Finally these predictions are **send to our AWS RDS Database** whom is get from our Django website to display it for our users with ```rds.py``` script.

This repository is pulled directly in our Data Achitecture on an **AWS EC2.**

```stop_instance.py``` script allows to the EC2 instance to **shutdown himself** with a AWS Lambda triggered by a AWS CloudWatch. 
