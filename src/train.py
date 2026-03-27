### 6. BUILD MODEL (ANN use Keras) ###
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)), # This is the input layer
    layers.Dense(32, activation="relu"), # This is first hidden layer with 32 neurons and "relu" activation
    layers.Dropout(0.4), # This layer will dropout 40% of the neurons during training. To anti-overfitting
    layers.Dense(16, activation="relu"), # This is second hidden layer with 16 neurons and "relu" activation
    layers.Dropout(0.3), # This layer will dropout 30% of the neurons during training. To anti-overfitting
    layers.Dense(1, activation="sigmoid"),  # This is the output layer with 1 neuron output and "sigmoid" activation
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3), # This will set optimizer to Adam optimizer with 0.001 learning rate.
    loss="binary_crossentropy", # To measure how wrong the model is.
    metrics=[
        keras.metrics.AUC(name="auc"),
        "accuracy"
    ]
)

model.summary()

### 7. TRAIN MODEL ###

es = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=16,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)

print("Training done.")

# put training history to variables
train_acc_history  = history.history["accuracy"]
val_acc_history    = history.history["val_accuracy"]
train_loss_history = history.history["loss"]
val_loss_history   = history.history["val_loss"]
val_auc_history    = history.history["val_auc"]

final_train_acc  = train_acc_history[-1]
final_val_acc    = val_acc_history[-1]
final_train_loss = train_loss_history[-1]
final_val_loss   = val_loss_history[-1]
best_val_auc     = max(val_auc_history)

print("==== Training summary ====")
print(f"Final Train Accuracy      : {final_train_acc:.4f}")
print(f"Final Validation Accuracy : {final_val_acc:.4f}")
print(f"Final Train Loss          : {final_train_loss:.4f}")
print(f"Final Validation Loss     : {final_val_loss:.4f}")
print(f"Best Validation AUC       : {best_val_auc:.4f}")

# Plot learning curves
import matplotlib.pyplot as plt

plt.plot(train_acc_history, label="train_acc")
plt.plot(val_acc_history, label="val_acc")
plt.xlabel("epoch"); plt.ylabel("accuracy")
plt.title("Accuracy curve")
plt.legend(); plt.tight_layout(); plt.show()

plt.plot(train_loss_history, label="train_loss")
plt.plot(val_loss_history, label="val_loss")
plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("Loss curve")
plt.legend(); plt.tight_layout(); plt.show()
