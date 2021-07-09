# fast-weights-test
Implementation of Using Fast Weights to Attend to the Recent Past with tensorflow2.0

```
Using Fast Weights to Attend to the Recent Past
Jimmy Ba, Geoffrey Hinton, Volodymyr Mnih, Joel Z. Leibo, Catalin Ionescu
NIPS 2016, https://arxiv.org/abs/1610.06258

```
## Requirement
* tensorflow (>2.0.0)

## Execution
* To see the advantage behind the fast weights, Ba et. al. used a very simple toy task: Given: g1o2k3??g we need to predict 1.

* Create datasets
```
python data_utils.py
```

* For training
```
python train.py
```

## Results
After about 100 epochs of training, the prediction accuracy reached 100%

## Extensions
* Use fast weight for language modeling
