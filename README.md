# Deep Leakage From Gradients [[arXiv]](https://arxiv.org/abs/1906.08935)
### Weight initialisation influence on data reconstruction via leakage from Gradients.

This project aim to analyse the influence of weight initialisation on data reconstruction via leakage from Gradients.

### **Launch multi-run based on Hydra Sweeper:**

```shell
python main.py -m --config-name=experiment
```
### **Launch single-run:**

```bash
python main.py -m <options>
```

> **model parameters**
>
> [ model ] = { McMahan_CNN, Custom_ResNet}
>
> [ model.init_method ] = { xavier\_normal\_, xavier\_uniform\_ , kaiming\_normal_, kaiming\_uniform\_ }
>
> [ model.activation ] = {ReLU, Sigmoid}
>
> ------
>
> **Attacker Parameters**
>
> [ attacker.optimizer.type ] = { Adam, LBFGS, SGD }
>
> [ attacker.optimizer.learning_rate ] = any float
>
> [ attacker.optimizer.scheduler ] = { True, False }
>
> [ attacker.loss_function ] = { MSE, cosine_similarity }
>
> ------
>
> **Client Parameters**
>
> [ client.prune.type ] = { random, small, None }
>
> [ client.prune.percentage ] = any float between 0-1 

