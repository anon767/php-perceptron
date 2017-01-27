<?php
require __DIR__ . '/vendor/autoload.php';
use NeuralNetwork\Network;
use NeuralNetwork\ActivationFunction\TanhFunction;


echo "<pre>";
/**
 * sqrt(n*m) neurons for three-layer hiddenlayer
 */
$n = new Network([4,5, 3], new TanhFunction());

$fileContents = str_replace("Iris-virginica", "-1,-1,1", str_replace("Iris-versicolor", "-1,1,-1", str_replace("Iris-setosa", "1,-1,-1", file_get_contents("TestData/IRIS.txt"))));

$testCases = explode("\n", $fileContents);

shuffle($testCases);
for ($i = 0; $i < 146; $i++) {
    $params = explode(",", $testCases[$i]);
    $n->train([$params[0], $params[1], $params[2], $params[3]], [$params[4], $params[5], $params[6]]);
}


echo "<hr>";
//5.1,3.8,1.5,0.3,Iris-setosa
$testCase = $n->calculate([5.1, 3.8, 1.5, 0.3]);
print_r("Iris Setosa: <br>");
print_r($testCase);
echo "<hr>";
//7.0,3.2,4.7,1.4,Iris-versicolor
$testCase = $n->calculate([7.0, 3.2, 4.7, 1.4]);
print_r("Iris Versicolor: <br>");
print_r($testCase);
echo "<hr>";
//7.2,3.0,5.8,1.6,Iris-virginica
$testCase = $n->calculate([7.2, 3.0, 5.8, 1.6]);
print_r("Iris Virginica: <br>");
print_r($testCase);
echo "<hr>";
$n->getStatus();