# Neural Network
PHP Multilayer Perceptron


## test

Running the PHP Multilayer Perceptron on the Iris Flower Dataset
https://archive.ics.uci.edu/ml/datasets/iris

1. `php composer.phar update`
2. `php test.php`


## usage

For a simple Neural Network with one hidden layer (three hidden neurons, one input and one output neuron) and Tanh-Activation-Function.

```php
require __DIR__ . '/vendor/autoload.php';
use NeuralNetwork\Network;
use NeuralNetwork\ActivationFunction\TanhFunction;
```

use it as follow

```php
$n = new Network([1,3 1], new TanhFunction());
```

for training use

`$n->trainData($trainingset, $iterations);`

and to calculate an output

`print_r($n->calculate([1]));`
