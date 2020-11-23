from keras.models import load_model
import numpy as np
import cv2
import pickle
from image_segmentation import segment_image
from neural_network import resize_to_fit

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


def solve_captcha(image):
    # Load up the model labels
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)

    # Load up the trained model
    model = load_model(MODEL_FILENAME)

    # We do not know the number of characters here
    chars = segment_image(image, -1)

    if len(chars) > 0:
        output = cv2.merge([image] * 3)
        predictions = []

        # Loop over the characters
        for bounding_box in chars:
            x, y, w, h = bounding_box
            # Extract the char from the input image
            char_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
            # Re-size the letter image to 60x60 pixels to match training data
            char_image = resize_to_fit(char_image, 60, 60)

            if char_image is not None:
                # Expand dimensions
                char_image = np.expand_dims(char_image, axis=2)
                char_image = np.expand_dims(char_image, axis=0)

                # Use the model to make a prediction
                prediction = model.predict(char_image)

                # Convert the encoded prediction to specific label
                label = lb.inverse_transform(prediction)[0]
                predictions.append(label)

                # draw the prediction on the output image
                cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
                cv2.putText(output, label, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Print captcha
        captcha_text = "".join(predictions)
        print("CAPTCHA is: {}".format(captcha_text))

        return output, captcha_text

    return None, ''
