# Model Predimmo

Model Predimmo repository's job is to **train a deep learning model and made prediction** on real estate adverts from Leboncoin available in our AWS RDS with ```main.py```

* ```main.py``` script call ```preprocessing.py``` script whom **pre-processing** our datasets from datagouv.fr.
* That allow ```model.py``` script to **train our model** whom is **saved** in our daily backup model in save folder.
* This **artifact model** is used in our ```predict.py``` script to made **predictions** on real estate adverts from Leboncoin.
* Finally these predictions are **send to our AWS RDS Database** whom is get from our Django website to display it for our users.

This repository is pulled directly in our Data Achitecture on an **AWS EC2.**

```stop_instance.py``` script allows to the EC2 instance to **shutdown himself** with a AWS Lambda triggered by a AWS CloudWatch. 
