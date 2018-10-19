# Kohonen Self-Organising Map

Self organising maps are dimensionality reduction tools primarily used for visualising data and spotting clustering patterns based on their intrinsic features. Most clustering algorithms rely on the user to specify the number of hierarchical clusters with which to organis the data, however the **Kohonen Self-Organising Map** (SOM) only requires paramater tuning based on training speed and intensity.

## Getting Started

To use the SOM tool you can clone the repository and import the class using a suitable Python IDE.

## Prerequisites:

* numpy==1.14.0
* matplotlib==2.0.2

## Example code

Setting up the training data which will be presented to the algorithm:

```
train_data = np.array([[0,0,1],[0,1,0],[1,0,0],[1,1,1],[0,0,0],[1,1,0],
[1,0,1],[0,0,1],[0,0,0.8],[0,0,0.91],[0,0,0.95],[0,0,0.949]]) 
```
Instantiating the SOM class:

```
som_maker = SOM() 
```

Training the algorithm on the data:

```
som = SOM.generate_SOM(som_maker,x_size=50,y_size=75,your_data=train_data,
initial_radius=100,number_of_iterations=100,initial_learning_rate=0.1)       
```

Evaluate data on the trained SOM:

```
tested_data = SOM.evaluate(som_maker,train_data,som)
```

View the evaluated data:

```
plt.figure()
plt.plot(tested_data,'k*')
```

Here is an examplar pre-trained SOM with evaluated datapoints:

![Trained SOM](https://github.com/SpaceMeerkat/Bespin/blob/master/Example_Images/Trained_SOM.png)

## Things the SOM tool can't do (yet):

*Impute missing data
*Operate with batch inputs
*Output a map of any user-specified-size of <3 dimensions (currently SOM only outputs a 2D map)




