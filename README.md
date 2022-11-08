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
# print_string(input)
# param: input:string
# return: void
# Prints the string input to console
def print_string(input):
    print(input)
```

<h3> Git Commit Messages </h3>

- **Title**: A succinct summary of what changes you made and where they were
  - "Fixed minor bugs in the foward-pass loop" – Good! 👍
  - “Made some changes” – Bad! 👎 DO NOT DO THIS! >:(
- **Description** (optional for small changes, _required for major commits_): Talk more in-depth about what you did, and what needs to be worked on next
```
- Tried out different activation functions, decided on RELU
- Need to plug function into training to see if results are improved
```
  
