# How to use tramdag.sample()

we look at an example with 3 nodes $x,z$ continous and $y$ binary (modeled as ordinal).

![tramdag](https://github.com/buehlpa/TramDag/blob/main/docs/images/Sampling_triangular_dag.png)

# Sampling

Sampling can be done with the function:

```
samples,latents= model.sample()
```

## Observational Sampling  L1 


For a  tramdag model one can sample from the observational distribution. The  `model.sample()` method follows the causal order of the DAG.

e.g. for a SCM given by

$x = h_{x}^{-1}(u_{x})$

$y = h_{y}^{-1}(u_{y}|x )$

$z = h_{z}^{-1}(u_{z}|x ,y)$


1. The $u_{x}$ is sampled from a standard logistic distribution and the corresponding  $x$ sample is determined, by taking the inverse of the transformation function $h$.
2. with the sampled $x$ and a latent sample for $u_{y}$  a $y$ is determined.

3. with the sampled $y$ and $x$ and a latent sample for $u_{z}$ , a $z$ is determined.

4. redo steps 1-3 for any amount of samples, default is 10'000 controlled by the argument `model.sample(num_samples=100)`


## Interventional Sampling L2 (do calculus)

An intervention on a causal graph tries to answer the question what happens to the distributions of the variables if any variable is set to a fixed value $\alpha$ e.g the answer to the question what would happen if all patients were given treatment A instead of the observed (some got A and others B).

Interventional sampling follows the same structure as the observational. Lets say we want to intervene on $y$ with interventional value $\alpha=2$ The variables which are intervened on can be defined via argument  `model.sample(do_interventions={'y':2})`

The given SCM stays the same except for $y$ being $\alpha$

$x = h_{x}^{-1}(u_{x})$

$y = \alpha$

$z = h_{z}^{-1}(u_{z}|x ,y)$

## Counterfactual Sampling L3


Counterfactual queries become a bit more involved it tries to quantify the unknowable e.g. what had happened if the specific patient took medicine A instead of B.
Sounds like magic if you ask me, but no one asks me.
This process involves 3 steps:

1. Abduction: determining the $u$ s
2. Action: Adapting the SCM
3. Prediciton: Getting the Counterfactual 


Lets make an example for the three steps with the observation:
$x=0.2,y=1,z=-0.3$  ($y$ is ordinal and  $x,z$ continous). 

We ask the couterfactual question what would have happened to $z$ and $y$ if $x$ was 0.5 instead of the observed 0.2.

### 1. Abduction
In this case we have to know for each observation their according latent state $u$ correspondig to the observed value, which we get by evaluationg the fitted transformation function at the position of the observed value. 

This can be done with the function:

`model.get_latent(df)` with a `df` containing observations of interest which must contain all variables (except for the nodes which are intervened , but it doesn't hurt if they are included).


`get_latent()` returns the according u's from the transformations:

for continuous variables this results in:

$u=h(x|pa(x)) $


for ordinal modelled variables the according $u$ is ambiguous and is itself a random variable, following a truncated standard logistic distribution  with the cutpoints $h(k-1|pa(x))$ and $h(k|pa(x))$.

therefore the get_latent function returns the according $u$ for continuous variables but for ordinal variables it returns `u_lower` and `u_upper` corresponding to 
 $h(k-1|pa(x))$ and
  $h(k|pa(x))$ .



we create the dataframe with just 1 observation `df_with_one_obs`
```
u_df= model.get_latent(df_with_one_obs)
```
the u_df contains the columns:
```
index,x,x_U,y,y_U_lower,y_U_upper,z,z_U
```


 we get the according $u_{x}$ , $u_{y}$ , $u_{z}$ via:

$u_{x} = h_{x}(x=0.2)$


$u_{y}$ $\in$ [ $h_{y}(0|x=0.2)$ , $h_{y}(1|x=0.2)$ )



$u_{z} = h_{z}(z=-0.3|x=0.2,y=1)$



### 2. Action

Set $x_{cf}$ = 0.5 because we ask the question what had happened to $z$ and $y$ if $x$ was 0.5 instead of the observed 0.2.

### 3. Prediction 
We use the modified model with $x$=0.5 to determine the counterfactual as follows:
Again we follow the causal order by starting with the source node:  $x_{cf}$ = 0.5.

Since $y$ is ordinal the corresponding counterfactual $y_{cf}$ can in general not be determined unambigously. Instead we show how to determine the probability distribution of the possible counterfactual values.
To do so we first update the transformatino function $h(y|x_{cf}$ and then sample $n$,  $u_{y}$ $\in$ [ $h_{y}(0|x=0.2)$ , $h_{y}(1|x=0.2)$ ) from a truncated standard logistic distribution with these cutpoints $h_{y}(0|x=0.2)$ , $h_{y}(1|x=0.2)$  from step 1 and determine the corresponding $n$ counterfactual values $y_{cf_{j}}$ , j $\in$ {1 ... n}.
Now we can just count the frequencies of the determined counterfactuals and divide them by $n$  which leaves us with a probability distribution for $y_{cf}$


To propagate that further down to $z$ we have to proceed as follows:
We first update the transformation function $h_{z}^{-1}(u_{z}|x_{cf}=0.5 ,y_{cf})$, $y_{cf}$ could be different for each $y_{cf_{j}}$ and we use the latent value from step 1 $u_{z}$ to get $n$ according $z_{cf_{j}} =h_{z}^{-1}(u_{z}|x_{cf}=0.5 ,y_{cf_{j}})$ , which we can again count how many times certain values occured, leaving us with a distribution for $z_{cf}$ aswell.

Finally for the counterfactual question asked in this example tramdag returns :  (just some madeup numbers here, say $n$=1000 can be controlled via the argument `num_cf_latents` ):

$x_{cf}$ = 0.5

for ordinal variables with only continuous parents:
probabilities for each class (here binary):

$y_{cf}$=(0.2,0.8)

For continous variables with at least one ordinal parent we just return the values and how many times they were drawn. e.g. 700 times 0.5 and 300 times -0.4.

$z_{cf}$=(0.5:700,-0.4:300)



### Conclusion 
To conclude if we have ordinal variables in the system and want to do counterfactual queries, all the descendant nodes have a probability distribution for the counterfactual.
