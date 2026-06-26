# CNN básica en C++

Este proyecto implementa una red neuronal convolucional sencilla en C++17.
La idea es tener una base clara para entender cómo fluye una imagen por una CNN:
convolución, activación, pooling, aplanamiento, capa densa y salida con softmax.

Por ahora todo corre en CPU. La estructura está separada por módulos para que
más adelante podamos agregar una versión en CUDA sin reescribir toda la red.

## Estructura

```text
.
|-- CMakeLists.txt
|-- examples/
|   `-- main.cpp
|-- include/cnn/
|   |-- sequential.hpp
|   |-- tensor.hpp
|   |-- backend/
|   |   `-- cpu_ops.hpp
|   `-- layers/
|       |-- conv2d.hpp
|       |-- dense.hpp
|       |-- flatten.hpp
|       |-- layer.hpp
|       |-- max_pool2d.hpp
|       |-- relu.hpp
|       `-- softmax.hpp
`-- src/
    |-- sequential.cpp
    |-- tensor.cpp
    |-- backend/
    |   `-- cpu_ops.cpp
    `-- layers/
        |-- conv2d.cpp
        |-- dense.cpp
        |-- flatten.cpp
        |-- max_pool2d.cpp
        |-- relu.cpp
        `-- softmax.cpp
```

## Arquitectura de la red

El ejemplo principal usa esta secuencia:

```text
Input -> Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> Softmax
```

Con una entrada de forma:

```text
[1, 1, 8, 8]
```

La salida final tiene forma:

```text
[1, 3]
```

Es decir, una muestra con tres probabilidades de clase.

## Compilación

```powershell
cmake -S . -B build
cmake --build build
.\build\Debug\cnn_demo.exe
```

En algunos generadores el ejecutable puede quedar directamente en:

```text
build/cnn_demo
```

## Nota sobre CUDA

Las capas llaman a funciones del backend CPU, ubicadas en `src/backend/`.
Cuando implementemos CUDA, podemos agregar un backend nuevo con kernels en GPU
manteniendo casi igual la interfaz de las capas.
