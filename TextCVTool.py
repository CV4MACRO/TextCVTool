import matplotlib.pyplot as plt
import json
import time
import os

OCR_HELPER_JSON_PATH  = r"TextCVHelper.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH)) 

custom_config = f'--oem 4 --psm 6'

from ProcessCheckboxes import crop_image_and_sort_format
from ProcessPDF import PDF_to_images, preprocessed_image
from TextExtraction import get_data_and_landmarks, get_wanted_text


def TextExtractionTool(path, save_path=r"C:\Users\CF6P\Desktop\cv_text\Evaluate", custom_config=custom_config):
    start_time = time.time()
    images = PDF_to_images(path)
    res_dict_per_image = {}
    for i, image in enumerate(images,1):
        print(f"\n ------------ Image {i} is starting. time  : {(time.time() - start_time)} -------------------")
        processed_image = preprocessed_image(image)
        format, cropped_image = crop_image_and_sort_format(processed_image)
        print(f"Image is cropped, format is {format}. time  : {(time.time() - start_time)}")
        OCR_data, landmarks_dict = get_data_and_landmarks(format, cropped_image, ocr_config=custom_config)
        print(f"Landmarks are found. time  : {(time.time() - start_time)}")
        OCR_and_text_full_dict = get_wanted_text(cropped_image, landmarks_dict, format, ocr_config=custom_config)
        res_dict_per_image[i] = OCR_and_text_full_dict
        print(f"Text is detected. time  : {(time.time() - start_time)}")
        save_image_resultats(OCR_and_text_full_dict, path, i, save_path = save_path)
        print(f"Saved. time  : {(time.time() - start_time)}")
    return res_dict_per_image
    

def save_image_resultats(landmark_text_dict, path, i, save_path):
    _, name = os.path.split(path)
    name, _ = os.path.splitext(name)
    save_path = os.path.join(save_path, f"{name}_{i}.jpg")
    
    fig, axs = plt.subplots(5, 2, figsize=(30,30))
    a, b = 0, 0
    for i, (zone, dict) in enumerate(landmark_text_dict.items()):
        text = dict["sequences"]
        
        axs[a, b].imshow(dict["searching_area"])
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
    