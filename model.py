import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV3Small

class CustomMobileNetV3(layers.Layer):
    def __init__(self, output_classes=128, name='CustomMobileNetV3'):
        super(CustomMobileNetV3, self).__init__(name=name)
        # Initialize MobileNetV3Small with include_top=False to customize the top layers
        self.base_model = MobileNetV3Small(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
        # Adding custom layers for feature adaptation
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(output_classes, activation='relu')
        self.reshape = layers.Reshape((1, 1, output_classes))  # For compatibility with SE blocks if needed
        
    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_pool(x)
        x = self.dense1(x)
        x = self.reshape(x)
        return x

class EyeModel(CustomMobileNetV3):
    def __init__(self, name='EyeModel'):
        # Output classes set to 128 for feature vector from eye images
        super(EyeModel, self).__init__(output_classes=128, name=name)

class LandmarkModel(layers.Layer):
    def __init__(self, name='LandmarkModel'):
        super(LandmarkModel, self).__init__(name=name)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.dense3 = layers.Dense(32, activation='relu')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class GazeTrackModel(Model):
    def __init__(self, name='GazeTrackModel'):
        super(GazeTrackModel, self).__init__(name=name)
        self.eye_model = EyeModel()
        self.landmark_model = LandmarkModel()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.dense3 = layers.Dense(2)  # Assuming the output is a 2D vector for gaze direction

    def call(self, inputs):
        leftEye, rightEye, lms = inputs
        l_eye_feat = self.flatten(self.eye_model(leftEye))
        r_eye_feat = self.flatten(self.eye_model(rightEye))
        lm_feat = self.landmark_model(lms)
        combined_feat = tf.concat([l_eye_feat, r_eye_feat, lm_feat], axis=1)
        x = self.dense1(combined_feat)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def summary(self):
        input1 = Input(shape=(128, 128, 3))
        input2 = Input(shape=(128, 128, 3))
        input3 = Input(shape=(8,))
        model = Model(inputs=[input1, input2, input3], outputs=self.call([input1, input2, input3]))
        return model.summary()
