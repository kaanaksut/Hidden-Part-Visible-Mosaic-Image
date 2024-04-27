import cv2
import numpy as np

def block_processing(image, block_size):
    blocks = []
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                blocks.append(block)
            else:
                print("Invalid block shape:", block.shape)
    return blocks

def calculate_features(blocks):
    features = []
    for block in blocks:
        red_mean = np.mean(block[:,:,0])
        green_mean = np.mean(block[:,:,1])
        blue_mean = np.mean(block[:,:,2])
        features.append([red_mean, green_mean, blue_mean])
    return features

def calculate_histogram(features):
    histograms = []
    for feature in features:
        histogram, _ = np.histogram(feature, bins=256, range=(0, 256))
        histograms.append(histogram)
    return histograms

def main():
    image1_path = 'a1.png'
    image2_path = 'b1.png'
    block_size = 5
    
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)


    blocks_image1 = block_processing(image1, block_size)
    blocks_image2 = block_processing(image2, block_size)

    
    features_image1 = calculate_features(blocks_image1)
    features_image2 = calculate_features(blocks_image2)


    histograms_image1 = calculate_histogram(features_image1)
    histograms_image2 = calculate_histogram(features_image2)

    
    similarity_measure = []
    for hist1, hist2 in zip(histograms_image1, histograms_image2):
        similarity = np.sum(np.abs(np.array(hist1) - np.array(hist2)))
        similarity_measure.append(similarity)

    # En benzer bloğu bul
    most_similar_block_index = np.argmin(similarity_measure)
    print("Most similar block index:", most_similar_block_index)

    
    stego_image = image1.copy()
    if most_similar_block_index < len(blocks_image1) and most_similar_block_index < len(blocks_image2):
        i = most_similar_block_index // (image1.shape[1] // block_size)
        j = most_similar_block_index % (image1.shape[1] // block_size)
        stego_block = blocks_image2[most_similar_block_index]
        stego_block_resized = cv2.resize(stego_block, (block_size, block_size))
        stego_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = stego_block_resized
        cv2.imwrite('stego_image.png', stego_image)
    else:
        print("Most similar block index is out of bounds.")
        
if __name__ == "__main__":
    main()
