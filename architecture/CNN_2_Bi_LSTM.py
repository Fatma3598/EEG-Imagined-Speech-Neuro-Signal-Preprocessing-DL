from tensorflow.keras.layers import (
    Input,
    BatchNormalization,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dropout,
    Dense
)
from tensorflow.keras.models import Model


def CNN_2_Bi_LSTM(
    input_shape=(32, 14),
    name="CNN_2_Bi_LSTM",
    cnn_units=128,
    lstm_1_units=256,
    lstm_2_units=256,
    fully_connected_units=128,
    n_classes=30
):
    """
    CNN-2-Bi-LSTM architecture for EEG-based imagined speech classification.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input EEG window (samples, channels).
    name : str
        Name of the model.
    cnn_units : int
        Number of filters in the convolutional layer.
    lstm_1_units : int
        Number of units in the first Bi-LSTM layer.
    lstm_2_units : int
        Number of units in the second Bi-LSTM layer.
    fully_connected_units : int
        Number of units in the fully connected layer.
    n_classes : int
        Number of output classes.

    Returns
    -------
    tensorflow.keras.Model
        CNN-2-Bi-LSTM model.
    """

    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)

    x = Conv1D(
        filters=cnn_units,
        kernel_size=10,
        strides=1,
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(2)(x)

    x = Bidirectional(
        LSTM(
            lstm_1_units,
            activation="tanh",
            return_sequences=True
        )
    )(x)

    x = Dropout(0.5)(x)

    x = Bidirectional(
        LSTM(
            lstm_2_units,
            activation="tanh",
            return_sequences=False
        )
    )(x)

    x = BatchNormalization()(x)

    x = Dense(
        fully_connected_units,
        activation="relu"
    )(x)

    x = Dropout(0.5)(x)

    outputs = Dense(
        n_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=name
    )

    return model
