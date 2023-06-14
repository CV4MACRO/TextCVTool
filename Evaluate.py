import os
import pandas as pd
import json 

from TextCVTool import TextExtractionTool
from JaroDistance import jaro_distance

OCR_HELPER_JSON_PATH  = r"TextCVHelper.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH)) 


def _condition_fuction(col, proposition, data):
    if data == "None":
        return None
    if col == "parasite_recherche":
        count = 0
        data = list(data)
        for GT_parasite in data:
            GT_parasite
            if GT_parasite not in proposition:
                count+=1
        if count == 0:
            return 2
        elif count == 1:
            return 1
        else:
            return 0
        
    else:
        data = str(data)
        if proposition == data:
            return 2
        if jaro_distance(proposition, data)>0.8:
            return 1
        else :
            return 0
        
def eval_text_extraction(path_to_eval, eval_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate",
                         result_name = "0_results", score_name = "0_score"):
    
    result_excel_path = os.path.join(eval_path, result_name+".xlsx")
    data_col = ["root_path", "file_name", "page_number", "full_name"]
    zones_col = list(OCR_HELPER["regions"].keys())
    
    if os.path.exists(result_excel_path):
        eval_df = pd.read_excel(result_excel_path)
    else :
        eval_df = pd.DataFrame(columns=data_col+zones_col) # All detected text for all zones

    res_dict_per_image = TextExtractionTool(path_to_eval)
    root_path, file_name = os.path.split(path_to_eval)
    for image, zone_dict in res_dict_per_image.items():
        full_name =  root_path+"_"+os.path.splitext(file_name)[0]+"_"+str(image)
        row = [root_path, file_name, image, full_name]
        for _, landmark_text_dict in zone_dict.items():
            row.append(landmark_text_dict["text"])
        eval_df.loc[len(eval_df)] = row
    
    eval_df.to_excel(result_excel_path, index=False)
    
def get_score(data_excel_path = r"C:\Users\CF6P\Desktop\cv_text\Data\0_data.xlsx",
               eval_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate", result_name ="0_results", score_name = "0_score"):
    result_excel_path = os.path.join(eval_path, result_name+".xlsx")
    score_excel_path = os.path.join(eval_path, score_name+".xlsx")
    data_df = pd.read_excel(data_excel_path) # The real value of the scan
    eval_df = pd.read_excel(result_excel_path)
    zones_col = list(OCR_HELPER["regions"].keys())
    print("Evaluation is starting")
    
    data_df.drop(columns=["sheet_format"])
    data_df = data_df[data_df["full_name"].isin(eval_df["full_name"])].reset_index()
    score_df= eval_df[eval_df["full_name"].isin(data_df["full_name"])].reset_index()
    missing_annot = list(eval_df[~eval_df["full_name"].isin(score_df["full_name"])]["full_name"])
    if len(missing_annot)!=0:
        print(f" /!\ {missing_annot} Rows are missing in data annotation /!\ ")

    for col in zones_col:
        apply_df = score_df[["full_name", col]].merge(data_df[["full_name", col]], how='inner', on=["full_name"])
        apply_df.columns = ["full_name", col+"_score", col+"_data"]
        score_df[col] = apply_df.apply(lambda x : _condition_fuction(col, x[col+"_score"], x[col+"_data"]), axis=1)

    score_df.to_excel(score_excel_path, index=False)
    print("Evaluation is done")
    return   

l = [r"C:\Users\CF6P\Desktop\cv_text\Data\scan1.pdf"]
#, r"C:\Users\CF6P\Desktop\cv_text\Data\scan2.pdf", r"C:\Users\CF6P\Desktop\cv_text\Data\scan3.pdf"]
for el in l:
    eval_text_extraction(el)

get_score(score_name="new")