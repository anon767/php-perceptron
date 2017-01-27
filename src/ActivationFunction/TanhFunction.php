<?php
namespace NeuralNetwork\ActivationFunction;
/**
 * Created by PhpStorm.
 * User: Tom
 * Date: 22.01.2017
 * Time: 17:19
 */
class TanhFunction implements ActivationFunction
{

    public function activate($value)
    {
        return tanh($value);
    }

    public function derivativeActivate($value)
    {
        $tanh = tanh($value);
        return 1.0 - $tanh * $tanh;
    }
}