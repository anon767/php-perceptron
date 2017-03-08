<?php
require __DIR__ . '/vendor/autoload.php';
use NeuralNetwork\Network;
use NeuralNetwork\ActivationFunction\TanhFunction;



$n = new Network([4, 4, 3], new TanhFunction());

$fileContents = str_replace("Iris-virginica", "-1,-1,1", str_replace("Iris-versicolor", "-1,1,-1", str_replace("Iris-setosa", "1,-1,-1", file_get_contents("TestData/IRIS.txt"))));

$testCases = explode("\n", $fileContents);

$trainingset = [];


for ($i = 0; $i < count($testCases); $i++) {
    $params = explode(",", $testCases[$i]);
    $trainingset[] = [$params[0], $params[1], $params[2], $params[3], $params[4], $params[5], trim($params[6])];
}

$n->trainData($trainingset, 5000);
/*$j = 0;
do {
    shuffle($testCases);
    for ($i = 0; $i < 80; $i++) {
        $params = explode(",", $testCases[$i]);
        $n->train([$params[0], $params[1], $params[2], $params[3]], [$params[4], $params[5], $params[6]]);
    }
    $j++;

} while ($j < 100);
*/

echo "-1,1,-1";
print_r($n->calculate([7.0, 3.2, 4.7, 1.4]));
echo "\r\n1,-1,-1";
print_r($n->calculate([5.1, 3.8, 1.5, 0.3]));
echo "\r\n-1,-1,1";
print_r($n->calculate([7.2, 3.0, 5.8, 1.6]));