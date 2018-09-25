# coding: utf-8
from test_layer_naive import *


apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = test_MulLayer()
mul_tax_layer = test_MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
print("apple_price:",int(apple_price))
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
print("dapple_price:",dapple_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)
