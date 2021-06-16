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

