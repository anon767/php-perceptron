<?php
namespace NeuralNetwork\ActivationFunction;
/**
 * Created by PhpStorm.
 * User: Tom
 * Date: 22.01.2017
 * Time: 17:19
 */
interface ActivationFunction
{
    public function activate($value);

    public function derivativeActivate($value);
}