import os
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm
import itertools

def save_training_plots(history, model_name, output_dir):
    """Save training accuracy and loss plots."""
    plt.rcParams["figure.figsize"] = (15, 8)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title(f'{model_name} - Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Valid'], loc='upper left')

    # Plot loss
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title(f'{model_name} - Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Valid'], loc='upper right')

    # Save figure
    plot_path = os.path.join(output_dir, f'{model_name}_accuracy_loss.png')
    plt.savefig(plot_path, dpi=500)
    plt.close()
    return plot_path

def save_history_to_json(history, model_name, output_dir):
    """Save training history to JSON file."""
    history_dict = {}
    for epoch in range(len(history.history['accuracy'])):
        history_dict[epoch] = {
            "accuracy": history.history['accuracy'][epoch],
            "val_accuracy": history.history['val_accuracy'][epoch],
            "loss": history.history['loss'][epoch],
            "val_loss": history.history['val_loss'][epoch]
        }

    history_file_path = os.path.join(output_dir, f'{model_name}_training_history.json')
    with open(history_file_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    return history_file_path

def evaluate_model(model, X_test, y_test, batch_size):
    """Evaluate the model and return predictions."""
    loss, accuracy = model.evaluate(X_test, y_test)
    Y_pred = model.predict(X_test, batch_size=batch_size)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(Y_pred, axis=1)
    oa = accuracy_score(y_true, y_pred)
    aa = np.mean([accuracy_score(y_true[y_true == i], y_pred[y_true == i]) for i in range(len(np.unique(y_true)))])
    kappa = cohen_kappa_score(y_true, y_pred)

    return loss, accuracy, oa, aa, kappa, y_true, y_pred

def perform_tta(model, X_test, tta_steps, batch_size):
    """Perform Test-Time Augmentation (TTA) and return averaged predictions."""
    predictions = []
    for _ in tqdm(range(tta_steps)):
        preds = model.predict(X_test, batch_size=batch_size, verbose=1)
        predictions.append(preds)
        gc.collect()
    return np.mean(predictions, axis=0)

def save_classification_report(y_true, y_pred, model_name, output_dir):
    """Save classification report to a text file."""
    report = classification_report(y_true, y_pred)
    report_path = os.path.join(output_dir, f'{model_name}_classification_report.txt')
    with open(report_path, 'w') as file:
        file.write("Classification Report:\n")
        file.write(report)
    return report_path

def plot_confusion_matrix(cm, classes, model_name, output_dir, normalize=False):
    """Plot and save confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=500)
    plt.close()
    return cm_path

def evaluate_and_save_all_metrics(model, X_test, y_test, tta_steps, batch_size, class_labels, model_name, output_dir):
    """Evaluate model and save all evaluation metrics."""
    Y_pred_tta = perform_tta(model, X_test, tta_steps, batch_size)
    y_true = np.argmax(y_test, axis=1)
    y_pred_tta = np.argmax(Y_pred_tta, axis=1)

    cm = confusion_matrix(y_true, y_pred_tta)
    cm_path = plot_confusion_matrix(cm, class_labels, model_name, output_dir)

    loss, accuracy, oa, aa, kappa, _, _ = evaluate_model(model, X_test, y_test, batch_size)
    report_path = save_classification_report(y_true, y_pred_tta, model_name, output_dir)

    metrics = {
        "loss": round(loss, 4),
        "accuracy": round(accuracy, 4),
        "overall_accuracy": round(oa, 4),
        "average_accuracy": round(aa, 4),
        "kappa": round(kappa, 4),
        "confusion_matrix_path": cm_path,
        "classification_report_path": report_path
    }
    return metrics
