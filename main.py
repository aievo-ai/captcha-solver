import glob
import os
import pathlib
import math

from treat_images import load_image
from generate_characters import generate_characters
from image_segmentation import segment_image
from solve_captcha import solve_captcha
import neural_network
import cv2

TRAINING_FOLDER = "training_data"
TEST_DATA = "test_data"
OUTPUT_FOLDER = "extracted_images"


def data_augmentation():
    print("Augmenting data")
    count_synth = []
    letter_num = []
    remaining = []
    # Sum the number of files generated for each char
    for char in range(10):
        path = os.path.join(OUTPUT_FOLDER, str(char))
        count_synth.append(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))
        remaining.append(1000-count_synth[char])
        letter_num.append(math.ceil(remaining[char]/count_synth[char]))

    # Load all segmented images
    extracted_images = glob.glob(os.path.join(OUTPUT_FOLDER, "**/*.png"), recursive=True)
    for (i, captcha_char) in enumerate(extracted_images):
        print("[INFO] augmenting image {}/{}".format(i + 1, len(extracted_images)))
        folder = pathlib.PurePath(captcha_char)

        iterations = letter_num[int(folder.parent.name)]
        if iterations > remaining[int(folder.parent.name)]:
            iterations = remaining[int(folder.parent.name)]
        remaining[int(folder.parent.name)] -= iterations
        if iterations > 0:
            img = cv2.imread(captcha_char)
            # Create novel images transforming img for iterations times
            augmented_chars = generate_characters(img, iterations)
            for augmented_img in augmented_chars:
                count_synth[int(folder.parent.name)] += 1
                # Save synthetic images
                filename = os.path.join(folder.parent, "{}_synth.png".format(str(count_synth[int(folder.parent.name)]).zfill(6)))
                cv2.imwrite(filename, augmented_img)


def segmentation():
    print("Segmenting data")
    image_qty = 0
    char_qty = 0
    image_files = glob.glob(os.path.join(TRAINING_FOLDER, "*"))
    counts = {}

    for (i, captcha_file) in enumerate(image_files):
        print("[INFO] processing image {}/{}".format(i + 1, len(image_files)))

        img = load_image(captcha_file)

        # Get the base filename without the extension ("0147.jpeg" as "0147)
        filename = os.path.basename(captcha_file)
        captcha_text = os.path.splitext(filename)[0]

        chars = segment_image(img, len(filename) - 5)

        if len(chars) > 0:
            image_qty += 1
            char_qty += len(chars)

            # Save out each letter as a single image
            for letter_bounding_box, letter_text in zip(chars, captcha_text):
                # Grab the coordinates of the letter in the image
                x, y, w, h = letter_bounding_box

                # Extract the letter from the original image with a 2-pixel margin around the edge
                letter_image = img[y - 2:y + h + 2, x - 2:x + w + 2]

                # Get the folder to save the image in
                save_path = os.path.join(OUTPUT_FOLDER, letter_text)

                # if the output directory does not exist, create it
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # write the letter image to a file
                count = counts.get(letter_text, 1)
                p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
                cv2.imwrite(p, letter_image)

                # increment the count for the current key
                counts[letter_text] = count + 1

    print(image_qty, char_qty)


def train_neural_network():
    print("Training neural network")
    (data, labels) = neural_network.load_training_data(OUTPUT_FOLDER)
    (training_dataset, validation_dataset, training_lab, validation_lab) = neural_network.configure_training_data(data, labels)
    network = neural_network.create_neural_network()
    neural_network.train_neural_network(network, training_dataset, validation_dataset, training_lab, validation_lab)
    neural_network.save_neural_network(network)


def get_captcha():
    print("Testing neural network")
    false_negative = 0
    true_positive = 0
    false_positive = 0
    # Read image and resize
    captcha_image_files = glob.glob(os.path.join(TEST_DATA, "*"))
    for (i, captcha_image_file) in enumerate(captcha_image_files):
        print("[INFO] testing image {}/{}".format(i + 1, len(captcha_image_files)))
        captcha = load_image(captcha_image_file)
        _, result = solve_captcha(captcha)
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]
        if result == '':
            false_negative += 1
        elif captcha_correct_text == result:
            true_positive += 1
        else:
            false_positive += 1

        print(true_positive, false_positive, false_negative,
              true_positive / (true_positive + false_positive + false_negative))

    precision = true_positive/(true_positive+false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = true_positive / (true_positive + false_positive + false_negative)
    f1_score = 2 * ((precision*recall)/(precision+recall))
    print("Precision:", precision)
    print("Recall:", recall)
    print("Accuracy:", accuracy)
    print("F1 score:", f1_score)


if __name__ == "__main__":
    segmentation()
    data_augmentation()
    train_neural_network()
    get_captcha()
