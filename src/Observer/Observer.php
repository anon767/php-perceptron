<?php
namespace NeuralNetwork\Observer;
/**
 * Created by PhpStorm.
 * User: Tom
 * Date: 22.01.2017
 * Time: 18:48
 */
interface Observer
{
    public function listen($event, $data);
}