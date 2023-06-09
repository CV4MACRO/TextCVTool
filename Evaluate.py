import os
import pandas as pd

from ProcessPDF_obs import TextExtractionTool
from JaroDistance import jaro_distance


def _condition_fuction(proposition, data):
    if data == "None":
        return None
    if type(data) != type([]):
        data = str(data)
        if proposition == data:
            return 2
        if jaro_distance(proposition, data)>0.8:
            return 1
        else :
            return 0
    
    else :
        count = 0
        for GT_parasite in data:
            if GT_parasite not in proposition:
                count+=1
        if count == 0:
            return 2
        elif count == 1:
            return 1
        else:
            return 0
        
def eval_text_extraction(path_to_eval, results_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate",
                         eval_excel_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate\0_results.xlsx"):    
    data_col = ["root_path", "file_name", "page_number", "full_name"]
    zones_col = ["N_d_echantillon", "date_de_prelevement", "nom", "type_de_lot", 
             "variete",	"N_de_lot", "parasite_recherche"]
    if os.path.exists(eval_excel_path):
        eval_df = pd.read_excel(eval_excel_path)
    else :
        eval_df = pd.DataFrame(columns=data_col+zones_col) # All detected text for all zones

    res_dict_per_image = TextExtractionTool(path_to_eval)
    root_path, file_name = os.path.split(path_to_eval)
    for image, zone_dict in res_dict_per_image.items():
        full_name =  root_path+"_"+os.path.splitext(file_name)[0]+"_"+str(image)
        row = [root_path, file_name, image, full_name]
        for zone, landmark_text_dict in zone_dict.items():
            row.append(landmark_text_dict["text"])
        eval_df.loc[len(eval_df)] = row
    
    eval_df.to_excel(eval_excel_path, index=False)
    
def get_score(data_excel_path = r"C:\Users\CF6P\Desktop\cv_text\Data\0_data.xlsx",
               results_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate", eval_excel_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate\0_results.xlsx"):
    
    data_df = pd.read_excel(data_excel_path) # The real value of the scan
    eval_df = pd.read_excel(eval_excel_path)
    zones_col = ["N_d_echantillon", "date_de_prelevement", "nom", "type_de_lot", 
             "variete",	"N_de_lot", "parasite_recherche"]
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
        score_df[col] = apply_df.apply(lambda x : _condition_fuction(x[col+"_score"], x[col+"_data"]), axis=1)

    score_df.to_excel(os.path.join(results_path, "0_score.xlsx"), index=False)
    print("Evaluation is done")
    return   

l = [r"C:\Users\CF6P\Desktop\cv_text\Data\scan1.pdf", r"C:\Users\CF6P\Desktop\cv_text\Data\scan2.pdf", r"C:\Users\CF6P\Desktop\cv_text\Data\scan3.pdf"]
# for el in l:
eval_text_extraction(l[0])

get_score()