<?php
namespace NeuralNetwork\ActivationFunction;
/**
 * Created by PhpStorm.
 * User: Tom
 * Date: 22.01.2017
 * Time: 17:31
 */
class FermiFunction implements ActivationFunction
{

    public function activate($value)
    {
        return (1.0 / (1.0 + exp(-$value)));
    }

    public function derivativeActivate($value)
    {
        $v = $this->activate($value);

        return $v * (1 - $v);
    }
}