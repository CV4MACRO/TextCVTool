import matplotlib.pyplot as plt
import json
import time
import os

OCR_HELPER_JSON_PATH  = r"TextCVHelper.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH)) 

from ProcessPDF import PDF_to_images, preprocessed_image, crop_and_rotate
from TextExtraction import get_data_and_landmarks, get_wanted_text, _get_area


def TextExtractionTool(path):
    start_time = time.time()
    images = PDF_to_images(path)
    res_dict_per_image = {}
    for i, image in enumerate(images,1):
        print(f"\nImage {i} is starting. time  : {(time.time() - start_time)}")
        processed_image = preprocessed_image(image)
        cropped_image = crop_and_rotate(processed_image)
        print(f"Image is cropped. time  : {(time.time() - start_time)}")
        OCR_data, landmarks_dict = get_data_and_landmarks(cropped_image)
        print(f"Landmarks are found. time  : {(time.time() - start_time)}")
        landmark_and_text_dict = get_wanted_text(cropped_image, landmarks_dict)
        res_dict_per_image[i] = landmark_and_text_dict
        print(f"Text is detected. time  : {(time.time() - start_time)}")
        save_image_resultats(cropped_image, landmark_and_text_dict, path, i, save_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate")
        print(f"Saved. time  : {(time.time() - start_time)}")
    return res_dict_per_image
    

def save_image_resultats(cropped_image, landmark_text_dict, path, i, save_path):
    _, name = os.path.split(path)
    name, _ = os.path.splitext(name)
    save_path = os.path.join(save_path, f"{name}_{i}.jpg")
    
    fig, axs = plt.subplots(5, 2, figsize=(30,30))
    a, b = 0, 0
    for i, (zone, dict) in enumerate(landmark_text_dict.items()):
        text = dict["text"]
        if len(dict["landmark"][0][1])>0:
            areas = []
            for j, box in enumerate(dict["landmark"]):
                if box[0] == "found":
                    relat_pos = OCR_HELPER["regions"][zone]["relative_position"][j]
                else : relat_pos = [[0,1], [0,1]]
                areas.append(_get_area(cropped_image, dict["landmark"][j][1], relat_pos, corr_ratio=1.1))
            y_min, y_max, x_min, x_max = sorted(areas, key=(lambda x: (x[1]-x[0])*(x[3]-x[2])))[-1]
            axs[a, b].imshow(cropped_image[y_min:y_max, x_min:x_max])
            if zone == "parasite_recherche":
                t1, t2, t3 = text[:int(len(text)/3)], text[int(len(text)/3):int(2*len(text)/3):], text[int(2*len(text)/3):]
                axs[a, b].set_title(f'{zone} : \n {t1} \n {t2} \n {t3}', size = 30)
            else :
                axs[a, b].set_title(f'{zone} : \n {text}', size = 30)
        a+=1
        if i == 3 : 
            a=0
            b=1
    plt.plot()
    fig.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    
    print("start")
    path = r"C:\Users\CF6P\Desktop\cv_text\Data\scan1.pdf"
    TextExtractionTool(path)
    