import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
import pdf2image
import json
from unidecode import unidecode
from datetime import datetime
import os
import time

# note : filter for json deg-> ° ; Case landmark is empty
from JaroDistance import jaro_distance

custom_config = r'--oem 3 --psm 6'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\CF6P\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
poppler_path = r"C:\Users\CF6P\Downloads\Release-23.01.0-0\poppler-23.01.0\Library\bin"

OCR_HELPER_JSON_PATH  = r"TextCVHelper.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH))  
        
def PDF_to_images(path):
    images = pdf2image.convert_from_path(path, poppler_path=poppler_path)
    return [np.array(image) for image in images]

def preprocessed_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return thresh

def _points_filter(points):
    """
    Get the endpoint along each axis
    """
    points[points < 0] = 0
    xpoints = sorted(points, key=lambda x:x[0])
    ypoints = sorted(points, key=lambda x:x[1])
    tpl_x0, tpl_x1 = xpoints[::len(xpoints)-1]
    tpl_y0, tpl_y1 = ypoints[::len(ypoints)-1]
    return tpl_y0[1], tpl_y1[1], tpl_x0[0], tpl_x1[0]

def crop_and_rotate(processed_image):
    """
    Crop and rotate the image thanks to contour detection
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(~processed_image, kernel, iterations=2)
    contours,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contour found : Crop is impossible, processed image is return.")
        return processed_image
    contour = sorted(contours, key=cv2.contourArea)[-1] # Select the biggest contour
    rect = cv2.minAreaRect(contour) # get the rotate rectangle that monimise area
    box = np.intp(cv2.boxPoints(rect))    
    # Rotate image
    angle = rect[2]
    if 45<=angle<=90: angle = angle-90 # Angle correction , should be improve (robustness)
    rows, cols = processed_image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1) # Rotation matrix
    img_rot = cv2.warpAffine(processed_image,M,(cols,rows))
    # rotate bounding box, then crop
    points = np.intp(cv2.transform(np.array([box]), M))[0] # Points of angles of the box afeter rotatio,
    y0, y1, x0, x1 = _points_filter(points) # get the biggest crop
    img_crop = img_rot[y0:y1, x0:x1]

    if img_crop.shape[0]*img_crop.shape[1] < 1200*900:
        print("The found crop seems to be to short, processed image is returned.")
        return processed_image
    # plt.imshow(img_crop, cmap="gray")
    # plt.axis('off')
    # plt.show()
    return img_crop

def _landmark_word_filter(sequence):
    """
    Aim to make sure the found sequence of word has no empty string at the start or the end

    Args:
        sequence (list of strings)

    Returns:
       filterd_sequence (list of stringd)
       start, end (ints) : number of empty string to remove
    """
    left, right = 0, len(sequence)-1
    while sequence[left].isspace() or len(sequence[left]) == 0:
        left+=1
    while sequence[right].isspace() or len(sequence[right]) == 0:
        right-=1
    return sequence[left:right], (left, len(sequence)-1-right)

def _find_landmarks_index(key_sentences, text): # Could be optimized
    """
    Detect if the key sentence is seen by the OCR.
    If it's the case return the index where the sentence can be found in the text returned by the OCR,
    else return an empty array
    Args:
        key_sentences (list) : contains a list with one are more sentences, each word is a string.
        text (list) : text part of the dict returned by pytesseract 

    Returns:
        res_indexes (list) : [[start_index, end_index], [empty], ...]   Contains for each key sentence of the landmark the starting and ending index in the detected text.
                            if the key sentence is not detected res is empty.
    """
    res_indexes = []
    for key_sentence in key_sentences: # for landmark sentences from the json
        res = [] # 
        distance_max = 0.80
        if "NÂ°" in key_sentence :
            key_sentence = list(map(lambda x : x.replace("NÂ°", "N°"),key_sentence)) #Correction of json format
        for i_key, key_word in enumerate(key_sentence): # among all words of the landmark
            for i_word, word in enumerate(text): 
                if key_word.lower() == word.lower(): # if there is a perfect fit in an word (Maybe should be softer but would take more time)
                    distance = jaro_distance("".join(key_sentence), "".join(text[i_word-i_key:i_word-i_key+len(key_sentence)])) # compute the neighborood matching
                    if distance > distance_max : # take the best distance
                        distance_max = distance
                        res = [i_word-i_key, i_word-i_key+len(key_sentence)] # Start and end indexes of the found key sentence
                              
        if len(res)>0:
            res_text =  text[res[0]:res[1]] 
            _, (start_removed, end_removed) = _landmark_word_filter(res_text)
            res[0], res[1] = res[0]+start_removed, res[1]-end_removed
        res_indexes.append(res) # Empty if not found
    return res_indexes

def get_data_and_landmarks(cropped_image):
    """
    Perform the OCR on the processed image, find the landmarks and make sure there are in the right area 
    Args:
        cropped_image (array)

    Returns:
        res_landmarks (dict) :  { zone : {
                                "landmark" = [[x,y,w,h], []],
                                "risk" = int
            }
        }
        The coordinate of box around the key sentences for each zone, empty if not found
        OCR_data (dict) : pytesseract returned dict
    """
    image_height, image_width = cropped_image.shape[:2]
    res_landmarks = {}
    # Search text on the whole image
    OCR_data =  pytesseract.image_to_data(cropped_image, output_type=pytesseract.Output.DICT)
    text = [unidecode(word) for word in OCR_data["text"]]
    for zone, key_points in OCR_HELPER["regions"].items(): 
        detected_indexes = _find_landmarks_index(key_points["key_sentences"], text)
        landmark_region = key_points["subregion"] # Area informations
        xmin, xmax = image_width*landmark_region[1][0],image_width*landmark_region[1][1]
        ymin, ymax = image_height*landmark_region[0][0],image_height*landmark_region[0][1]
        landmarks_coord = []
        null = 0 # Relay if an OCR on a biggest area is needed
        for indexes in detected_indexes:
            if len(indexes)!=0 :
                i_min, i_max = indexes
                x, y = OCR_data['left'][i_min], OCR_data['top'][i_min]
                w = abs(OCR_data['left'][i_max-1] - x + OCR_data['width'][i_max-1])
                h = abs(int(np.mean(np.array(OCR_data['height'][i_min:i_max]))))
                if xmin<x<xmax and ymin<y<ymax: # Check if the found landmark is in the right area
                    landmarks_coord.append(("found", [x,y,w,h]))
                    res_landmarks[zone] = {"risk" : 0}
                    # cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else : 
                    landmarks_coord.append(("default", []))
                    null+=1
            else : 
                landmarks_coord.append(("default", []))
                null+=1
                
        # No detected landark: Let's search landmark on a smaller region  
        if null>0:
            OCR_default_region = pytesseract.image_to_data(cropped_image[int(ymin):int(ymax), int(xmin):int(xmax)], output_type=pytesseract.Output.DICT)
            for i, coord in enumerate(landmarks_coord):
                if len(coord[1])==0:
                    detected_index = _find_landmarks_index(key_points["key_sentences"][i], OCR_default_region)[0]
                    if len(detected_index)!=0 :
                        i_min, i_max = detected_index
                        x, y = OCR_default_region['left'][i_min] + int(xmin), OCR_default_region['top'][i_min]+int(ymin)
                        w = OCR_default_region['left'][i_max-1] - x + OCR_default_region['width'][i_max-1]
                        h = int(np.mean(np.array(OCR_default_region['height'][i_min:i_max])))
                        landmarks_coord[i] = ("found", [x,y,w,h])
                        res_landmarks[zone] = {"risk" : 0}
                    else : 
                        landmarks_coord[i] = ("default", [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)])
                        res_landmarks[zone] = {"risk" : 1}
        res_landmarks[zone]["landmark"] = landmarks_coord
    # plt.imshow(cropped_image, cmap="gray")
    # plt.show()
    return OCR_data, res_landmarks

def _get_area(cropped_image, box, relative_position, corr_ratio=1.1):
    """
    Get the area around the landmark box according to the given position
    Args:
        box (list): detected landmark box [x,y,w,h]
        relative_position ([[vertical_min,vertical_max], [horizontal_min,horizontal_max]]): number of box height and width to go to search the tet
    """
    im_y, im_x = cropped_image.shape[:2]
    x,y,w,h = box
    h_relative, w_relative = h*(relative_position[0][1]-relative_position[0][0])//2, w*(relative_position[1][1]-relative_position[1][0])//2
    y_mean, x_mean = y+h*relative_position[0][0]+h_relative, x+w*relative_position[1][0]+w_relative
    x_min, x_max = max(x_mean-w_relative*corr_ratio,0), min(x_mean+w_relative*corr_ratio, im_x)
    y_min, y_max = max(y_mean-h_relative*corr_ratio, 0), min(y_mean+h_relative*corr_ratio, im_y)
    (y_min, x_min) , (y_max, x_max) = np.array([[y_min, x_min], [y_max, x_max]]).astype(int)[:2]
    return y_min, y_max, x_min, x_max

def get_candidate_local_OCR(cropped_image, landmark_boxes, relative_positions):
    candidate_sequences = []
    for n_landmark, relative_position in enumerate(relative_positions):
        box_type, box = landmark_boxes[n_landmark]
        if box_type == "default" : relative_position = [[0,1], [0,1]]
        y_min, y_max, x_min, x_max = _get_area(cropped_image, box, relative_position)
        local_OCR = pytesseract.image_to_data(cropped_image[y_min:y_max, x_min:x_max], output_type=pytesseract.Output.DICT)
        sequence = []
        for i, word in enumerate(local_OCR["text"]):
            if not(word.isspace() or len(word) == 0 or word=="|"):
                sequence.append(word)
            elif len(sequence) != 0: # If space : new sequence
                if sequence not in candidate_sequences : # If it's a new sequence
                    candidate_sequences.append(sequence)
                sequence=[]
            if i == len(local_OCR["text"])-1 and len(sequence)!=0: # Add last sequence
                if max([jaro_distance("".join(sequence), "".join(candidate)) for candidate in candidate_sequences]) < 0.90 : # Skip similar sequence
                    candidate_sequences.append(sequence)
    return candidate_sequences

def _list_process(check_word, sequence_res):
    for candidate_sequence in sequence_res:
        for word in candidate_sequence:
            if jaro_distance(word.lower(), check_word.lower())>0.8 :
                return True

def _after_key_process(key, candidate_sequence, end_character=[":", "(*)"]):
    if any([key in unidecode(word) for word in candidate_sequence]):
        end_character.append(key)
        for character in end_character:
            for i, word in enumerate(candidate_sequence): # Traverse from end to beginning 
                if unidecode(word) == character :
                        return candidate_sequence[i+1:]
                try:
                    if unidecode(word)[-len(character)] == character : 
                        return candidate_sequence[i+1:]
                except IndexError:
                    pass
    return []

def condition_filter(candidate_sequences, key_main_sentence, conditions):
    strip_string_after_key = " |\[]-_!.<>{}—"
    strip_string_others = " |\/[]_!.<>{}()*:—"
    sequence_res = []
    
    for candidate_sequence in  candidate_sequences:
        res_candidate_sequence =[]
        for word in candidate_sequence:
            if len(word)>0 :
                if "after_key" not in [condition[0] for condition in conditions]:
                    if any(c.isalnum() for c in unidecode(word)):
                        if word not in key_main_sentence:
                            res_candidate_sequence.append(word.strip(strip_string_others))
                else :
                    res_candidate_sequence.append(word.strip(strip_string_after_key))       
        sequence_res.append(res_candidate_sequence)
        
    for condition in conditions:
        new_sequence = []
        
        if condition[0] == "after_key": 
            key = condition[1]
            end_character=[":", "(*)"] # Order is important
            for candidate_sequence in sequence_res: # Detected sequences wich need iteration over themselves
                found_sequence = _after_key_process(key, candidate_sequence, end_character)
                new_sequence.append(found_sequence)
                
        if condition[0] == "date":
            date_format = "%d/%m/%Y"
            for candidate_sequence in sequence_res: # Detected sequences wich need iteration over themselves
                candidate_sequence = [word.strip(strip_string_others) for word in candidate_sequence]
                for word in candidate_sequence :
                    try:
                        date = bool(datetime.strptime(word, date_format))
                    except ValueError:
                        date = False
                    if date==True : 
                        new_sequence.append(word) # Extract the right word among others of the sequence
                    
        if condition[0] == "start":
            for candidate_sequence in sequence_res: # Detected sequences wich need iteration over themselves
                for i, word in enumerate(candidate_sequence) :
                    start = condition[1]
                    if word[:len(start)] == start : new_sequence.append("".join(candidate_sequence[i:]))
                        
        if condition[0] == "list": # In this case itertion is over element in the condition list
            check_list = condition[1]
            for check_elmt in check_list:
                check_words = check_elmt.split(" ")
                count = 0
                for check_word in check_words:
                    if _list_process(check_word, sequence_res):
                        count+=1
                    if count >= len(check_words) and check_elmt not in new_sequence: 
                        new_sequence.append(check_elmt)
        new_sequence = [seq for seq in new_sequence if len(seq)>0]
        sequence_res = new_sequence
        
    return sequence_res

def select_text(clean_text): # More case by case function
    if len(clean_text) ==0 :
            return clean_text
    elif len(clean_text) ==1 :
        if type(clean_text[0]) == type([]):
            return " ".join(clean_text[0])
        else:
            return clean_text[0]
    else: # The list as more than one proposition (lists or strings)
        if type(clean_text[0]) == type([]): # Mutiple propositions from different conditions
                for i in range(len(clean_text)):
                    for j in range(i+1,len(clean_text)):
                        if jaro_distance("".join(clean_text[i]), "".join(clean_text[j]))>0.5 : # if the content of the two lists is the same return
                            return " ".join([word for word in clean_text[i] if word in clean_text[j]])
                return " ".join(clean_text[0]) # Else, arbitrary retrun the first text

        if type(clean_text[0]) == type(""):
                for i in range(len(clean_text)):
                    for j in range(i+1,len(clean_text)):
                        if clean_text[i]==clean_text[j] :
                            return " ".join([clean_text[i]]) # Clean the list form double
        return clean_text

def common_mistake_filter(condition_text, zone):
    clean_text = []
    if zone == "nom":
        for sequence in condition_text:
            new_sequence = []
            for i, word in enumerate(sequence):
                word = ''.join([i for i in word if not i.isdigit()])
                if word.lower()[:5] == "eurof" :
                    new_sequence = sequence[:i]
                    break
                elif len(word)>0 : new_sequence.append(word)
            clean_text.append(new_sequence)
    else : clean_text = condition_text
    return clean_text
     
def get_wanted_text(cropped_image, landmarks_dict):
    res_dict = landmarks_dict.copy()
    for zone, key_points in OCR_HELPER["regions"].items():
        landmark_boxes =  landmarks_dict[zone]["landmark"]
        candidate_sequences = get_candidate_local_OCR(cropped_image, landmark_boxes, key_points["relative_position"])
        condition_text = condition_filter(candidate_sequences, key_points["key_sentences"][0], key_points["conditions"])
        cleaned_text = common_mistake_filter(condition_text, zone)
        res_text = select_text(cleaned_text) # Normalize and process condition text (ex : Somes are simple lists other lists of lists...)
        res_dict[zone]["text"] = res_text
        if zone in ["N_de_lot", "localisation_prelevement", "N_de_scelle"] :
            print("\n", zone)
            print(candidate_sequences)
            print("condition_text :", condition_text)
    return res_dict 

def TextExtractionTool(path):
    start_time = time.time()
    images = PDF_to_images(path)
    images = [images[2], images[4]]
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
        save_image_resultats(cropped_image, landmark_and_text_dict, save_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate"+ f"\\scan1_{i}.jpg")
        print(f"Saved. time  : {(time.time() - start_time)}")
    return res_dict_per_image
    

def save_image_resultats(cropped_image, landmark_text_dict, save_path):
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

if __name__ == "__main__":
    
    print("start")
    path = r"C:\Users\CF6P\Desktop\cv_text\Data\scan1.pdf"
    TextExtractionTool(path)
    
# Fixed json N°   
# Maked "Nom" detection more robust

# Scan3_5 NOM