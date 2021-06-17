Tatodet for language detection/classification 
====

This is a standalone python microservice to classify text into language classes using models trained on tatoeba's data 

Requirements
-----

Mainly pymc3, numpy, and sklearn. However many other packages are used and many more will be added in the future.

```sh
pip install -r requirements.txt
```


Models implemented
----

- Heirarchical Bayesian model:
 - Beta with freq priors
 - Beta with Poisson priors

Webapi
---

Make sure the model is built:


```sh
python3 build_model.py
```

To run the webapi:

```sh
python3 api.py
```

then send a query:


```sh
curl -X GET "http://localhost:8080/v1/det?sent=what+is+it&trials=20"
```

Tests
---

To run tests:

```sh
python3 -m pytest
```
