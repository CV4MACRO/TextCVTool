import os
import pandas as pd
import json 
from datetime import date
import pytesseract
today = str(date.today().strftime("%b-%d-%Y"))

from TextCVTool import TextExtractionTool
from JaroDistance import jaro_distance
custom_config = f'--oem 4 --psm 6'
OCR_HELPER_JSON_PATH  = r"TextCVHelper.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH)) 


def _condition_fuction(col, proposition, data):
    proposition = str(proposition)
    if data == "None":
        return None
    if col == "parasite_recherche":
        proposition = list(proposition)
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
                         result_name = "0_results", custom_config=custom_config):
    
    result_excel_path = os.path.join(eval_path, result_name+".xlsx")
    data_col = ["root_path", "file_name", "page_number", "full_name"]
    zones_col = list(OCR_HELPER["hand"].keys())+["format"]
    
    if os.path.exists(result_excel_path):
        eval_df = pd.read_excel(result_excel_path)
    else :
        eval_df = pd.DataFrame(columns=data_col+zones_col) # All detected text for all zones

    res_dict_per_image = TextExtractionTool(path_to_eval, custom_config=custom_config)
    root_path, file_name = os.path.split(path_to_eval)
    for image, zone_dict in res_dict_per_image.items():
        full_name =  root_path+"_"+os.path.splitext(file_name)[0]+"_"+str(image)
        row = [root_path, file_name, image, full_name]
        for _, landmark_text_dict in zone_dict.items():
            row.append(landmark_text_dict["sequences"])
        row.append(landmark_text_dict["format"])
        eval_df.loc[len(eval_df)] = row
    
    eval_df.to_excel(result_excel_path, index=False)
    
def get_score(result_name, eval_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate",
              data_excel_path = r"C:\Users\CF6P\Desktop\cv_text\Data\0_data.xlsx", score_name = "1_score"):
    
    result_excel_path = os.path.join(eval_path, result_name+".xlsx")
    score_excel_path = os.path.join(eval_path, score_name+".xlsx")
    data_df = pd.read_excel(data_excel_path) # The real value of the scan
    eval_df = pd.read_excel(result_excel_path)
    zones_col = list(OCR_HELPER["hand"].keys())
    print("Evaluation is starting")
    
    data_df.drop(columns=["sheet_format"])
    data_df = data_df[data_df["full_name"].isin(eval_df["full_name"])].reset_index()
    score_df= eval_df[eval_df["full_name"].isin(data_df["full_name"])].reset_index()
    missing_annot = list(eval_df[~eval_df["full_name"].isin(score_df["full_name"])]["full_name"])
    if len(missing_annot)!=0:
        print(f" /!\ {missing_annot} Rows are missing in data annotation /!\ ")

    for col in zones_col:
        print(col)
        apply_df = score_df[["full_name", col]].merge(data_df[["full_name", col]], how='inner', on=["full_name"])
        apply_df.columns = ["full_name", col+"_score", col+"_data"]
        score_df[col] = apply_df.apply(lambda x : _condition_fuction(col, x[col+"_score"], x[col+"_data"]), axis=1)

    print(score_df[zones_col].stack().value_counts())
    score_df.to_excel(score_excel_path, index=False)
    print("Evaluation is done")
    return score_df[zones_col].stack().value_counts()

if __name__ == "__main__":

    eval_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate\conf"
    l = [r"C:\Users\CF6P\Desktop\cv_text\Data\scan1.pdf", r"C:\Users\CF6P\Desktop\cv_text\Data\scan2.pdf", r"C:\Users\CF6P\Desktop\cv_text\Data\scan3.pdf"]
    
    for i in [0, 1, 3, 6, 11, 12, 13]:
        for j in [0, 3]:
    # l = [r"C:\Users\CF6P\Desktop\cv_text\Data\scan2.pdf"]
    # for i in [1]:
    #     for j in [0, 3]:
            conf = f'--oem {j} --psm {i}'
            result_name = f"0_results{i}_{j}"
            stack = []
            try :
                for el in l:
                    eval_text_extraction(el, eval_path=eval_path, result_name=result_name, custom_config=conf)
                stack.append([(i,j), get_score(result_name=result_name, eval_path=eval_path, score_name=f"1_score_{today}")])
                with open(r"C:\Users\CF6P\Desktop\cv_text\Evaluate\conf\stack.txt", 'a') as f:
                    f.write(str(stack))
            except pytesseract.pytesseract.TesseractError:
                print("BUG TESSERACT : ", i, j)
                pass
            except FileNotFoundError:
                print("FILE : ", i, j )
                pass
    # get_score(result_name= f"0_results_.xlsx", score_name=f"1_score_{today}")