<h1> epai-sentiment-of-tweets </h1>

<h2> Project Code Style Guide </h2>

<h3> Naming Convention </h3>

- **Class Names**: Snake_Case
  - Ex. RNN_Forward_Pass, Main_Window
- **Function Names**: snake_case
  - Ex. loss_function, plot_validation_curve
- **Variable Names**: camelCase
  - Ex. batchSize, enableGPU

<h3> Commenting Convention </h3>

_Please make it a habit to also comment all of your variables, calculations, etc. It’s either that or spending extra time to explain to your teammates what every piece does,,, and then comment them 🥴_

- **Functions and Classes** commenting format:
```
Function/Class name
Parameters and their corresponding types
Return variables and their corresponding types
Brief description of what the function does
```

Example:
```
# print_string(string input)
# param: input:string
# return: void
# Prints the string input to console
def print_string(input):
    print(input)
```
