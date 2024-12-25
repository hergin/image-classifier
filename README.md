Images are 800x800 in all of them except noted in simpler model.

### Most sophisticated model (10 epoch)
```
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])
```
* It took more than 2 hours and 3.7GB with accuracy 0.9921

### Simpler model (10 epoch)
```
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(*img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])
```
* It took 30 mins and 1.9GB with accuracy 0.9979
* This model with 5 epoch and 300x300 images took only 1 minute and 266MB with accuracy 0.9911

### Simpler model less dense (5 epoch)
```
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(*img_size, 3)),  # Reduced to 8 filters
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(32, activation='relu'),  # Reduced dense layer size
    Dense(train_data.num_classes, activation='softmax')
])
```
* It took 10 mins and 477MB with accuracy 0.9922

### Simpler model separable convolution (5 epoch)
```
model = Sequential([
    SeparableConv2D(8, (3, 3), activation='relu', input_shape=(*img_size, 3)),  # Separable convolution
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])
```
* It took 8 mins and 477MB with accuracy 0.8692

### Simpler model less filter less dense (5 epoch)
```
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(*img_size, 3)),  # Reduced to 8 filters
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(32, activation='relu'), # Reduced dense layer size
    Dense(train_data.num_classes, activation='softmax')
])
```
* It took 8 mins and 477MB with accuracy 0.9780