import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import json
from copy import deepcopy
from unidecode import unidecode

import locale
locale.setlocale(locale.LC_TIME,'')
from datetime import datetime

from ProcessCheckboxes import crop_image_and_sort_format, get_format_or_checkboxes, get_lines, Template
from ProcessPDF import PDF_to_images, preprocessed_image
from JaroDistance import jaro_distance

custom_config = f'--oem 3 --psm 6'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\CF6P\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

OCR_HELPER_JSON_PATH  = r"TextCVHelper.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH))

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

def get_data_and_landmarks(format, cropped_image, JSON_HELPER=OCR_HELPER, ocr_config=custom_config):
    """
    Perform the OCR on the processed image, find the landmarks and make sure there are in the right area 
    Args:
        cropped_image (array)

    Returns:
        res_landmarks (dict) :  { zone : {
                                "landmark" = [[x,y,w,h], []],
            }
        }
        The coordinate of box around the key sentences for each zone, empty if not found
        OCR_data (dict) : pytesseract returned dict
    """
    image_height, image_width = cropped_image.shape[:2]
    res_landmarks = {}
    # Search text on the whole image
    OCR_data =  pytesseract.image_to_data(cropped_image, output_type=pytesseract.Output.DICT, config = ocr_config)
    text = [unidecode(word) for word in OCR_data["text"]]
    for zone, key_points in JSON_HELPER[format].items(): 
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
                    # cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else : 
                    landmarks_coord.append(("default", []))
                    null+=1
            else : 
                landmarks_coord.append(("default", []))
                null+=1
                
        # No detected landmark: Let's search landmark on a smaller region  
        if null>0:
            OCR_default_region = pytesseract.image_to_data(cropped_image[int(ymin):int(ymax), int(xmin):int(xmax)], 
                                                           output_type=pytesseract.Output.DICT, config = custom_config)
            for i, coord in enumerate(landmarks_coord):
                if len(coord[1])==0:
                    detected_index = _find_landmarks_index(key_points["key_sentences"][i], OCR_default_region)[0]
                    if len(detected_index)!=0 :
                        i_min, i_max = detected_index
                        x, y = OCR_default_region['left'][i_min] + int(xmin), OCR_default_region['top'][i_min]+int(ymin)
                        w = OCR_default_region['left'][i_max-1] - x + OCR_default_region['width'][i_max-1]
                        h = int(np.mean(np.array(OCR_default_region['height'][i_min:i_max])))
                        landmarks_coord[i] = ("found", [x,y,w,h])
                    else : 
                        landmarks_coord[i] = ("default", [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)])
        res_landmarks[zone] = {"landmark" : landmarks_coord}
    # plt.imshow(cropped_image, cmap="gray")
    # plt.show()
    return OCR_data, res_landmarks

def _process_raw_text_to_sequence(OCR_text):
    candidate_sequences = [] # Store all sequence
    candidate_indexes = []
    sequence = [] # Stack a chunck of non-separated word
    indexes = []
    for i, word in enumerate(OCR_text):
            if not(word.isspace() or len(word) == 0 or word=="|"):
                sequence.append(word)
                indexes.append(i)
            elif len(sequence) != 0: # If space : new sequence
                if sequence not in candidate_sequences:
                    indexes.append(i)
                    candidate_indexes.append(indexes)
                    candidate_sequences.append(sequence)
                sequence=[]
                indexes=[]
            if i == len(OCR_text)-1 and len(sequence)!=0: # Add last sequence
                if sequence not in candidate_sequences:
                    candidate_sequences.append(sequence)
                    indexes.append(i+1)
                    candidate_indexes.append(indexes)

    return candidate_sequences, candidate_indexes

def _get_area(cropped_image, box, relative_position, corr_ratio=1.1):
    """
    Get the area coordinates of the zone thanks to the landmark and the given relative position
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

def get_candidate_local_OCR(cropped_image, landmark_boxes, relative_positions, format, ocr_config=custom_config):
    OCRs_and_candidates_list = []
    for n_landmark, relative_position in enumerate(relative_positions):
        res_dict = {}
        box_type, box = landmark_boxes[n_landmark]
        if box_type == "default" : 
            relative_position = [[0,1], [0,1]]
        y_min, y_max, x_min, x_max = _get_area(cropped_image, box, relative_position)
        searching_area = cropped_image[y_min:y_max, x_min:x_max]
        # plt.imshow(searching_area)
        # plt.show()
        local_OCR = pytesseract.image_to_data(searching_area, output_type=pytesseract.Output.DICT, config=ocr_config)
        candidate_sequences, candidate_indexes = _process_raw_text_to_sequence(local_OCR["text"])
        res_dict["OCR"], res_dict["type"], res_dict["landmark_box"] = local_OCR, box_type, box
        res_dict["searching_area"] = searching_area
        res_dict["sequences"], res_dict["indexes"] = candidate_sequences, candidate_indexes
        res_dict["risk"] = 0
        res_dict["format"] = format
        if res_dict["type"] == "default":
            res_dict["risk"] = 1
        OCRs_and_candidates_list.append(res_dict)
    
    check_seq = []
    for i_candidate, candidate_dict in enumerate(OCRs_and_candidates_list): # delete double sequence
        res_seq, res_index = [], []
        for i_seq, seq in enumerate(candidate_dict["sequences"]):
            if not seq in check_seq:
                check_seq.append(seq)
                res_seq.append(seq)
                res_index.append(candidate_dict["indexes"][i_seq])
        if res_seq == []:
            del OCRs_and_candidates_list[i_candidate]
        else:
            OCRs_and_candidates_list[i_candidate]["sequences"], OCRs_and_candidates_list[i_candidate]["indexes"] = res_seq, res_index
    return OCRs_and_candidates_list

def _list_process(check_word, candidate_sequence, candidate_index):
    for i_word, word in enumerate(candidate_sequence):
        if jaro_distance(word.lower(), check_word.lower())>0.8:
            return True, candidate_index[i_word]
    return False, None

def _after_key_process(key, candidate_sequence, cleaned_index, similarity=0.9):
    for i_word, word in enumerate(candidate_sequence):
        if jaro_distance(key, unidecode(word).strip('()*:;,"'))>similarity:
            if i_word < len(candidate_sequence)-1:
                following_seq, following_id = candidate_sequence[i_word+1:], cleaned_index[i_word+1:]
                following_seq[0] = following_seq[0].lstrip('()*:;,"')
                return following_seq, following_id
        try:
            if unidecode(word).strip('()*:;,"')[-len(key):] == key : # If the end_character is not a single word but a the end of a string
                following_seq, following_id = candidate_sequence[i_word+1:], cleaned_index[i_word+1:]
                following_seq[0] = following_seq[0].lstrip('()*:;,"')
                return following_seq, following_id
            if unidecode(word).strip('()*:;,"')[:len(key)] == key : # If the word is not a single word but a the start of a string
                following_word = unidecode(word).strip('()*:;,"')[len(key):].lstrip('()*:;,"')
                following_seq, following_id = candidate_sequence[i_word+1:], cleaned_index[i_word+1:]
                if following_word != '':
                    following_seq = [following_word] + following_seq
                    following_id = [i_word] + following_id
                return following_seq, following_id
        except IndexError:
            pass      
    
    return [], []

def _clean_local_sequences(sequence_index_zips, key_main_sentences, conditions):
    strip_string_after_key = " |\[]_!'.<>{}—;"
    strip_string_others = "()*: |\/[']_!.<>{}—;-"
    cleaned_candidate, cleaned_indexes = [], []
    key_sentences = [word for sentence in key_main_sentences for word in sentence]
    for candidate_sequence, candidate_indexes in sequence_index_zips:
        res_candidate_sequence, res_candidate_indexes = [], []
        for i_word, word in enumerate(candidate_sequence):
            if "after_key" not in [condition[0] for condition in conditions]:
                if any(c.isalnum() for c in unidecode(word)):
                    if word not in key_sentences:
                        res_candidate_sequence.append(word.strip(strip_string_others))
                        res_candidate_indexes.append(candidate_indexes[i_word])
            else:
                res_candidate_sequence.append(word.strip(strip_string_after_key))
                res_candidate_indexes.append(candidate_indexes[i_word])
                    
        cleaned_candidate.append(res_candidate_sequence)
        cleaned_indexes.append(res_candidate_indexes)
    return cleaned_candidate, cleaned_indexes

def condition_filter(candidates_dicts, key_main_sentences, conditions):
    OCRs_and_candidates = deepcopy(candidates_dicts)
    OCRs_and_candidates_filtered = []
    for candidate_dict in OCRs_and_candidates:
        strip_string_others = "()* |\/[]_!.<>{}:—;-"
        zipped_seq = zip(candidate_dict["sequences"], candidate_dict["indexes"])
        clean_sequence, clean_indexes = _clean_local_sequences(zipped_seq, key_main_sentences, conditions)
        zipped_seq = zip(clean_sequence, clean_indexes)
        for condition in conditions:
            new_sequence, new_indexes = [], []
            if condition[0] == "after_key": 
                key = condition[1]
                for candidate_sequence, candidate_index in zipped_seq: # Detected sequences wich need iteration over themselves
                    # print("after key :", candidate_sequence)
                    found_sequence, found_index = _after_key_process(key, candidate_sequence, candidate_index)
                    new_sequence.append(found_sequence)
                    new_indexes.append(found_index)
                    
            if condition[0] == "date": # Select a date format
                date_formats = ["%d/%m/%Y"] # + ["%d %B %Y"]
                for candidate_sequence, candidate_index in zipped_seq:
                    for i_word, word in enumerate(candidate_sequence):
                        word = word.lower().strip(strip_string_others+"abcdefghijklmnopqrstuvwxyz") # Could be cleaner 
                        for date_format in date_formats:
                            try:
                                _ = bool(datetime.strptime(word, date_format))
                                new_sequence.append([word])
                                new_indexes.append([candidate_index[i_word]])
                            except ValueError:
                                pass
                        
            if condition[0] == "echantillon": # A special filter for numero d'echantillon
                NUM1, NUM2 = condition[1] # Thresholds
                for candidate_sequence, candidate_index in zipped_seq: # Detected sequences wich need iteration over themselves
                    for i_word, word in enumerate(candidate_sequence):
                        try :
                            if NUM1 < int(word[:len(str(NUM1))]) < NUM2 : # Try to avoid strings shorter than NUM
                                new_sequence.append(["".join(candidate_sequence[i_word:])])
                                new_indexes.append(candidate_index[i_word:])
                        except ValueError:
                            pass
                            
            if condition[0] == "list": # In this case itertion is over element in the condition list
                concat_seq, concat_index = [], []
                for candidate_sequence, candidate_index in zipped_seq:
                    concat_seq+=candidate_sequence
                    concat_index+= candidate_index
                check_list = condition[1]
                for check_elmt in check_list:
                    check_indexes=[]
                    check_words = check_elmt.split(" ")
                    for check_word in check_words:
                        status, index = _list_process(check_word, concat_seq, concat_index)
                        if status:
                            check_indexes.append(index)
                        if len(check_indexes) == len(check_words) and check_elmt not in new_sequence: # All word of the checking elements are in the same candidate sequence
                            new_sequence.append([check_elmt])
                            new_indexes.append(check_indexes)
                sorted_by_id = sorted(zip(new_sequence, new_indexes), key=lambda x: x[1][0])
                if sorted_by_id != []:
                    new_sequence , new_indexes = zip(*sorted_by_id)
                    
            new_sequence_res, new_indexes_res = [], []            
            for i in range(len(new_sequence)):
                if new_sequence[i] != []:
                    new_sequence_res.append(new_sequence[i])
                    new_indexes_res.append(new_indexes[i])
            zipped_seq = zip(new_sequence_res, new_indexes_res)
            
        candidate_dict["sequences"], candidate_dict["indexes"] = new_sequence_res, new_indexes_res
        OCRs_and_candidates_filtered.append(candidate_dict)
    return OCRs_and_candidates_filtered

def _get_checkbox_word(sequence, index, ref_word_list, strips = ["1I 1[]JL_-|", "CI 1()[]JL_-|"]): # Several strips to be careful with word starting with C or I
    for i_word, word in enumerate(sequence):
        for strip in strips:
            word = word.strip(strip)
            if word in ref_word_list:
                return [word], [index[i_word]]
    return [], []
                
def get_checkbox_check_format(format, checkbox_dict, cropped_image, landmark_boxes, relative_positions):
    OCRs_and_candidates_list = []
    for n_landmark, relative_position in enumerate(relative_positions):
        res_dict = {}
        box_type, box = landmark_boxes[n_landmark]
        if box_type == "default" : relative_position = [[0,1], [0,1]]
        y_min, y_max, x_min, x_max = _get_area(cropped_image, box, relative_position)
        searching_area = cropped_image[y_min:y_max, x_min:x_max]
        templates = [Template(image_path=checkbox_dict["cross_path"], label="cross", color=(0, 0, 255), matching_threshold=0.5)]
        checkboxes = get_format_or_checkboxes(searching_area, mode="get_boxes", TEMPLATES=templates, show=False)
        sorted_checkboxes = sorted([checkbox for checkbox in checkboxes if checkbox["LABEL"]=="cross"], key=lambda obj: obj["MATCH_VALUE"], reverse=True)
        res_dict["OCR"] = {}
        res_dict["type"], res_dict["box"] = box_type, box
        res_dict["sequences"], res_dict["indexes"] = [], []
        res_dict["searching_area"] = searching_area
        res_dict["risk"] = 0
        res_dict["format"] = format
        for cross in sorted_checkboxes:
            x1,y1, x2,y2= cross["TOP_LEFT_X"], cross["TOP_LEFT_Y"], cross["BOTTOM_RIGHT_X"], cross["BOTTOM_RIGHT_Y"]
            h, w = abs(y2-y1), abs(x2-x1)
            end_x = min(x_max, x2+2*w)
            y1, y2 = max(int(y1-h/2),0) , min(y_max,int(y2+h/2))
            new_area = searching_area[y1:y2, x2:end_x]
            sequence = []
            while (end_x-x2)<10*w :
                local_OCR = pytesseract.image_to_data(new_area, output_type=pytesseract.Output.DICT, config = '--oem 3 --psm 8')
                sequence, index = _process_raw_text_to_sequence(local_OCR["text"])
                if len(sequence)>0: 
                    res_word, res_index = _get_checkbox_word(sequence[0], index[0], checkbox_dict["list"]) # POSTULATE : sequence is contains only one list
                    if res_word != []:
                        res_dict["OCR"] = local_OCR
                        res_dict["sequences"], res_dict["indexes"] = [res_word], [res_index]
                        OCRs_and_candidates_list.append(res_dict)
                        break
                end_x += w
                end_x = min(x_max, end_x)
                new_area = searching_area[y1:y2, x2:end_x]
            
            if res_dict["sequences"] != []: # Stop the iteration over crosses 
                break
        if OCRs_and_candidates_list == []:
            OCRs_and_candidates_list.append(res_dict)
        
    return OCRs_and_candidates_list
    
def get_checkbox_table_format(checkbox_dict, clean_OCRs_and_candidates):
    OCRs_and_candidates_list = []
    templates = [Template(image_path=checkbox_dict["cross_path"], label="cross", color=(0, 0, 255), matching_threshold=0.4)]        
    for candidate_dict in clean_OCRs_and_candidates:
        res_dict = candidate_dict
        searching_area = candidate_dict["searching_area"]
        checkboxes = get_format_or_checkboxes(searching_area, mode="get_boxes", TEMPLATES=templates, show=False)
        sorted_checkboxes = sorted([checkbox for checkbox in checkboxes if checkbox["LABEL"]=="cross"], key=lambda obj: obj["MATCH_VALUE"], reverse=True)
        
        parasite_location = []
        for parasite, indexes in zip(candidate_dict["sequences"], candidate_dict["indexes"]):
            parasite_dict =  {}
            parasite_dict["parasite"], parasite_dict["indexes"] = parasite, indexes
            last_index = indexes[-1]
            parasite_dict["top"], parasite_dict["height"] = res_dict["OCR"]["top"][last_index], res_dict["OCR"]["height"][last_index]
            parasite_location.append(parasite_dict)
        
        parasite_list, parasite_index = [], []
        for checkbox in sorted_checkboxes:
            top_mid_bottom = [checkbox["TOP_LEFT_Y"], (checkbox["TOP_LEFT_Y"]+checkbox["BOTTOM_RIGHT_Y"])/2, checkbox["BOTTOM_RIGHT_Y"]]
            distance_list = []
            found = False
            for parasite_dict in parasite_location:
                if any([parasite_dict["top"]<point<(parasite_dict["top"]+parasite_dict["height"]) for point in top_mid_bottom]): # Can select multiple choices
                    parasite_list.append(parasite_dict["parasite"])
                    parasite_index.append(parasite_dict["indexes"])
                    found = True
                elif not found:
                    distance_list.append((abs(parasite_dict["top"]-top_mid_bottom[0]), parasite_dict))
            if not found and distance_list!= []:
                nearest_para = min(distance_list, key=lambda x: x[0])
                parasite_list.append(nearest_para[1]["parasite"])
                parasite_index.append(nearest_para[1]["indexes"])
        
        res_dict["sequences"]  = parasite_list
        res_dict["indexes"] = parasite_index
        OCRs_and_candidates_list.append(res_dict)
    return OCRs_and_candidates_list    

def common_mistake_filter(OCRs_and_candidates, zone):
    if zone == "nom": # 
        clean_OCRs_and_candidates = []
        for candidate_dict in OCRs_and_candidates:
            res_seq, res_index = [], []
            sequences, indexes = candidate_dict["sequences"], candidate_dict["indexes"]
            for sequence, index in zip(sequences, indexes):
                new_sequence = sequence
                new_index = index
                for i, word in enumerate(sequence):
                    if word.lower()[:5] == "eurof":
                        new_sequence = sequence[:i]
                        new_index = index[:i]
                        break
                res_seq.append(new_sequence)
                res_index.append(new_index)
            candidate_dict["sequences"], candidate_dict["indexes"] = res_seq, res_index
            clean_OCRs_and_candidates.append(candidate_dict)
            return clean_OCRs_and_candidates
    else :
        return OCRs_and_candidates

def select_text(OCRs_and_candidates, zone): # More case by case function COULD BE IMPROVE WITH RECURSIVITY
    final_OCRs_and_text_dict = []
    strip = " |\_!.<>{}—;'-"
    for candidate_dict in OCRs_and_candidates:
        # Get sequence as simple string ON A LIST (not as a list of strings)
        res_seq, res_index = [], [] 
        sequences, indexes = candidate_dict["sequences"], candidate_dict["indexes"] # POSTULATE : is sequences contains only 1 type
        if sequences==[] or (len(sequences)==1 and sequences[0]==[]):
            res_seq, res_index = [[]], [[]]
        if len(sequences) ==1 :
            if type(sequences[0]) == type([]): # The only one elmnt is a list
                kept_i = [i for i in range(len(sequences[0])) if sequences[0][i] not in strip] # Select non-strip indices only 
                res_seq.append([" ".join([sequences[0][i].strip(" ") for i in kept_i]).strip(strip)])
                res_index = [[indexes[0][i] for i in kept_i]]
            else:
                print("rare case 1")
                res_seq.append([sequences[0].strip(strip)])
                res_index = indexes
        if len(sequences)>1 : # The list as more than one proposition (lists or strings)
            if type(sequences[0]) == type(""): # Elements are strings
                print("rare case 2")
                kept_i = [i for i in range(len(sequences)) if sequences[i] not in strip]
                res_seq.append([" ".join([sequences[i].strip(" ") for i in kept_i]).strip(strip)])
                res_index = [indexes[i] for i in kept_i]
                
            if type(sequences[0]) == type([]): # Mutiple propositions as list
                kept_i = [[i for i in range(len(seq_list)) if seq_list[i] not in strip] for seq_list in sequences]
                res_seq = [[" ".join([sequences[i_block][j_word].strip(" ") for j_word in kept_i[i_block]]).strip(strip)] for i_block in range(len(kept_i))]
                res_index = [[indexes[i_block][j_word] for j_word in kept_i[i_block] ]for i_block in range(len(kept_i))]
        
        if zone == "parasite_recherche":
            candidate_dict["sequences"], candidate_dict["indexes"] = [res_seq], res_index # Trick to keep all element of the list
        else:
            candidate_dict["sequences"], candidate_dict["indexes"] = res_seq, res_index
        final_OCRs_and_text_dict.append(candidate_dict)
        
    if len(final_OCRs_and_text_dict) == 1: # If ther is only one dict
        if len(final_OCRs_and_text_dict[0]["sequences"]) == 1: # With 1 option
            final_OCRs_and_text_dict[0]["choice"] = "one choice"
            final_OCRs_and_text_dict[0]["sequences"] = final_OCRs_and_text_dict[0]["sequences"][0]
            final_OCRs_and_text_dict[0]["indexes"] = final_OCRs_and_text_dict[0]["indexes"][0]
            return final_OCRs_and_text_dict[0]
        else: # With two or more options
            final_OCRs_and_text_dict[0]["sequences"] = final_OCRs_and_text_dict[0]["sequences"][0]
            final_OCRs_and_text_dict[0]["indexes"] = final_OCRs_and_text_dict[0]["indexes"][0]
            final_OCRs_and_text_dict[0]["risk"] += 1 # Multple option, one is chosen arbitary
            final_OCRs_and_text_dict[0]["choice"] = "One dict with Multiple choice, first one is taken (risk +1)"
            return final_OCRs_and_text_dict[0]
    else:
        # Searching for matching case
        for i_dict in range(len(final_OCRs_and_text_dict)):
            for j_dict in range(i_dict+1, len(final_OCRs_and_text_dict)):
                for i_range in range(len(final_OCRs_and_text_dict[i_dict]["sequences"])):
                    i_sequence, i_index = final_OCRs_and_text_dict[i_dict]["sequences"][i_range], final_OCRs_and_text_dict[i_dict]["indexes"][i_range]
                    for j_range in range(len(final_OCRs_and_text_dict[j_dict]["sequences"])):
                        j_sequence, j_index = final_OCRs_and_text_dict[j_dict]["sequences"][j_range], final_OCRs_and_text_dict[j_dict]["indexes"][j_range]
                        if i_sequence == j_sequence: # Multiple choice but two of them are excatly the same
                            final_OCRs_and_text_dict[i_dict]["sequences"] = i_sequence
                            final_OCRs_and_text_dict[i_dict]["indexes"] = i_index
                            final_OCRs_and_text_dict[i_dict]["choice"] = "Multiple dict but same text"
                            return final_OCRs_and_text_dict[i_dict]
                            
                        if jaro_distance("".join(i_sequence), "".join(j_sequence))>0.8: # Similar answer, give confidence to found landamrk data
                            if final_OCRs_and_text_dict[i_dict]["type"] == "found":
                                final_OCRs_and_text_dict[i_dict]["sequences"] = i_sequence
                                final_OCRs_and_text_dict[i_dict]["indexes"] = i_index
                                final_OCRs_and_text_dict[i_dict]["choice"] = "Multiple dict with similar text and found landmark"
                                return final_OCRs_and_text_dict[i_dict]
                            if final_OCRs_and_text_dict[j_dict]["type"] == "found":
                                final_OCRs_and_text_dict[j_dict]["sequences"] = j_sequence
                                final_OCRs_and_text_dict[j_dict]["indexes"] = j_index
                                final_OCRs_and_text_dict[j_dict]["choice"] = "Multiple dict with similar text and found landmark"
                                return final_OCRs_and_text_dict[j_dict]
                            else : 
                                final_OCRs_and_text_dict[i_dict]["sequences"] = i_sequence # No found landmark, res is chosen arbitrary
                                final_OCRs_and_text_dict[i_dict]["indexes"] = i_index
                                final_OCRs_and_text_dict[i_dict]["choice"] = "Multiple dict with similar text but no found landmark (risk+1)"
                                return final_OCRs_and_text_dict[i_dict]
            
            # Else
        for i_dict in range(len(final_OCRs_and_text_dict)):
            i_sequence, i_index = final_OCRs_and_text_dict[i_dict]["sequences"][i_range], final_OCRs_and_text_dict[i_dict]["indexes"][i_range]
            #No matching with other proposition, selected the one which seems the more "accurate"
            if final_OCRs_and_text_dict[i_dict]["type"] == "found" and len(final_OCRs_and_text_dict[i_dict]["sequences"])==1: 
                final_OCRs_and_text_dict[i_dict]["sequences"] = i_sequence
                final_OCRs_and_text_dict[i_dict]["indexes"] = i_index
                final_OCRs_and_text_dict[i_dict]["choice"] = "Multiple dict with different text but a found landmark as an unique text"
                return final_OCRs_and_text_dict[i_dict]
            
            if len(final_OCRs_and_text_dict[i_dict]["sequences"])==1: 
                final_OCRs_and_text_dict[i_dict]["sequences"] = i_sequence
                final_OCRs_and_text_dict[i_dict]["indexes"] = i_index
                final_OCRs_and_text_dict[i_dict]["risk"] += 1
                final_OCRs_and_text_dict[i_dict]["choice"] = "Multiple dict with different text but a default landmark has an unique text"
                return final_OCRs_and_text_dict[i_dict]
            
        final_OCRs_and_text_dict[i_dict]["sequences"] = final_OCRs_and_text_dict[0]["sequences"][0] 
        final_OCRs_and_text_dict[i_dict]["indexes"] = final_OCRs_and_text_dict[0]["indexes"][0]
        final_OCRs_and_text_dict[i_dict]["risk"] += 1
        final_OCRs_and_text_dict[i_dict]["choice"] = "Multiple different choices, the first element of the first dict is arbitrary taken"
        
    
        return final_OCRs_and_text_dict[i_dict]

def get_wanted_text(cropped_image, landmarks_dict, format, JSON_HELPER=OCR_HELPER, ocr_config=custom_config):
    res_dict_per_zone = {}
    for zone, key_points in JSON_HELPER[format].items():
        # print(f"\n NEW ZONE - {zone} :")
        landmark_boxes =  landmarks_dict[zone]["landmark"]
        conditions =  key_points["conditions"]
        if (format, zone) == ("check", "type_lot"):
            checkbox_dict = JSON_HELPER["checkbox"][format][zone]
            candidate_OCR_list_filtered =  get_checkbox_check_format(format, checkbox_dict, cropped_image, landmark_boxes, key_points["relative_position"])
        else:
            candidate_OCR_list = get_candidate_local_OCR(cropped_image, landmark_boxes, key_points["relative_position"], format, ocr_config=ocr_config)
            candidate_OCR_list_filtered = condition_filter(candidate_OCR_list, key_points["key_sentences"], conditions)
            
        # for d in candidate_OCR_list:
        #     print("first candidate : ", d["sequences"])
       
        clean_OCRs_and_candidates = common_mistake_filter(candidate_OCR_list_filtered, zone)
        
        if (format, zone) == ("table", "parasite_recherche"):
            checkbox_dict = JSON_HELPER["checkbox"][format][zone]
            clean_OCRs_and_candidates = get_checkbox_table_format(checkbox_dict, clean_OCRs_and_candidates)
        # print(clean_OCRs_and_candidates[0]["indexes"])
        # print(clean_OCRs_and_candidates[0]["sequences"])
        OCR_and_text_full_dict = select_text(clean_OCRs_and_candidates, zone) # Normalize and process condition text (ex : Somes are simple lists other lists of lists...)
        
        if OCR_and_text_full_dict["sequences"] != [] and zone != "parasite_recherche":
             OCR_and_text_full_dict["sequences"] =  OCR_and_text_full_dict["sequences"][0] # extract the value
        if OCR_and_text_full_dict["indexes"] != [] :
            # print(OCR_and_text_full_dict["indexes"])
            if type(OCR_and_text_full_dict["indexes"][0]) == type([]):
                OCR_and_text_full_dict["indexes"] = OCR_and_text_full_dict["indexes"][0]
            
        res_dict_per_zone[zone] = OCR_and_text_full_dict

        print("seq : ", OCR_and_text_full_dict["sequences"])
        # print("choice : ", OCR_and_text_full_dict["choice"])
        # print("index : ", OCR_and_text_full_dict["indexes"], "\n")
        
        # for i in OCR_and_text_full_dict["indexes"]:
        #     try :
        #         print(OCR_and_text_full_dict["OCR"]["text"][i], " : ", OCR_and_text_full_dict["OCR"]["conf"][i])
        #     except TypeError:
        #         for j in i:
        #             print(OCR_and_text_full_dict["OCR"]["text"][j], " : ", OCR_and_text_full_dict["OCR"]["conf"][j])    
                    
    return res_dict_per_zone 


if __name__ == "__main__":
    
    print("start")
    path = r"C:\Users\CF6P\Desktop\cv_text\Data\scan3.pdf"
    images = PDF_to_images(path)
    images = images[4:]
    res_dict_per_image = {}
    for i, image in enumerate(images,1):
        print(f"\n -------------{i}----------------- \nImage {i} is starting")
        processed_image = preprocessed_image(image)
        format, cropped_image = crop_image_and_sort_format(processed_image)
        print(f"Image with format : {format} is cropped.")
        OCR_data, landmarks_dict = get_data_and_landmarks(format, cropped_image)
        print(f"Landmarks are found.")
        landmark_and_text_dict = get_wanted_text(cropped_image, landmarks_dict, format)
        res_dict_per_image[i] = landmark_and_text_dict
    