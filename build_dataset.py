import os 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from dataclasses import dataclass 
from pathlib import Path 
from jqdatasdk import *



@dataclass 
class CFG :
    LOCAL_ROOT :str ="./local_data"


    ALPHA101_DIR :str ="../local_data/get_all_alpha_101/daily"
    ALPHA191_DIR :str ="../local_data/get_all_alpha_191/daily"
    JQF_DIR :str ="../local_data/get_factor_values/daily"


    PRIVATE_DIR :str ="../private_factor_library"


    INDUSTRY_DIR :str ="../local_data/get_industry"


    TARGET_DIR :str ="../labels"




def _read_parquet_if_exists (path :Path )->pd .DataFrame :
    if not path .exists ():
        return pd .DataFrame ()
    df =pd .read_parquet (path )
    return df 


def _load_factor_day (path :Path ,prefix :str ,*,date :str ,has_date_col :bool =True )->pd .DataFrame :

    df =_read_parquet_if_exists (path )
    if df .empty :
        return pd .DataFrame (columns =["code"])

    df =df .copy ()
    if "code"not in df .columns :
        return pd .DataFrame (columns =["code"])

    df ["code"]=df ["code"].astype (str )

    if has_date_col and "date"in df .columns :

        try :
            d0 =pd.to_datetime (date ).date()
            dd =pd.to_datetime (df ["date"],errors ="coerce").dt .date 
            df =df.loc[dd ==d0 ]
        except Exception :
            pass 


    df =df .dropna (subset =["code"]).drop_duplicates ("code",keep ="last")


    drop_cols ={"date"}
    feat_cols =[c for c in df .columns if c not in ({"code"}|drop_cols )]
    out =df [["code"]+feat_cols ].copy ()


    ren ={c :f"{prefix}{c}"for c in feat_cols }
    out =out .rename (columns =ren )

    return out 


def _winsorize (df :pd .DataFrame ,q :float =0.01 )->pd .DataFrame :

    if df .empty :
        return df 
    lo =df .quantile (q )
    hi =df .quantile (1 -q )
    return df .clip (lower =lo ,upper =hi ,axis =1 )


def _get_industry_label_long (ind_path :Path ,codes :pd .Index )->pd .Series :

    ind =_read_parquet_if_exists (ind_path )
    if ind .empty or ("code"not in ind .columns ):
        return pd .Series ("UNK",index =codes ,dtype ="object")

    ind =ind .copy ()
    ind ["code"]=ind ["code"].astype (str )


    need_cols =["code","type","industry_code"]
    for c in need_cols :
        if c not in ind .columns :
            return pd .Series ("UNK",index =codes ,dtype ="object")
    ind =ind [need_cols ].dropna (subset =["code","industry_code"])


    type_priority =["sw_l1","jq_l1","sw_l2","jq_l2","sw_l3","zjw"]
    types =ind ["type"].astype (str )
    chosen =None 
    for t in type_priority :
        if (types ==t ).any ():
            chosen =t 
            break 
    if chosen is None :
        chosen =types .value_counts ().index [0 ]

    ind =ind .loc [types ==chosen ].copy ()

    ind =ind .drop_duplicates ("code",keep ="last").set_index ("code")["industry_code"].astype (str )

    out =ind .reindex (codes )
    out =out .fillna ("UNK")
    return out 


def _industry_median_fill (X :pd .DataFrame ,ind_label :pd .Series )->pd .DataFrame :

    if X .empty :
        return X 
    g =ind_label .reindex (X .index ).fillna ("UNK")
    med =X .groupby (g ).transform ("median")
    return X .fillna (med ).fillna (X .median (numeric_only =True ))


def _neutralize_by_industry_and_size (
X :pd .DataFrame ,
ind_label :pd .Series ,
ln_mcap :pd .Series ,
*,
exclude_cols =None ,
)->pd .DataFrame :

    if X .empty :
        return X 

    exclude_cols =exclude_cols or []
    cols =[c for c in X .columns if c not in set (exclude_cols )]
    if not cols :
        return X .copy ()

    X0 =X [cols ].astype (float )


    g =ind_label .reindex (X0 .index ).fillna ("UNK").astype (str )
    G =pd .get_dummies (g ,prefix ="ind",dummy_na =False ).astype (float )
    G =G .loc [:,G .std (ddof =0 )>0 ]


    s =ln_mcap .reindex (X0 .index ).astype (float )
    s =s .replace ([np .inf ,-np .inf ],np .nan )
    s =s .fillna (s .median ())
    s =s .values .reshape (-1 ,1 )


    if G .shape [1 ]==0 :
        Z =s 
    else :
        Z =np .column_stack ([G .values ,s ])



    Z_pinv =np .linalg .pinv (Z )
    B =Z_pinv @X0 .values 
    Xn0 =X0 .values -(Z @B )

    Xn =pd .DataFrame (Xn0 ,index =X0 .index ,columns =X0 .columns )


    out =X .copy ()
    out .loc [:,cols ]=Xn .loc [:,cols ]
    return out 


def _zscore (df :pd .DataFrame )->pd .DataFrame :
    if df .empty :
        return df 
    mu =df .mean ()
    sd =df .std (ddof =0 )
    keep =sd >0 
    out =df .loc [:,keep ].copy ()
    out =(out -mu [keep ])/sd [keep ]
    return out .astype ("float32")


def _pick_market_cap_column (df :pd .DataFrame ):

    if df is None or df .empty :
        return None 
    cols =list (df .columns )
    if "jq_market_cap"in cols :
        return "jq_market_cap"


    end_caps =[c for c in cols if c .endswith ("market_cap")]

    if end_caps :
        return end_caps [0 ]


    any_caps =[c for c in cols if "market_cap"in c ]
    return any_caps [0 ]if any_caps else None 







def build_range (start_date :str ,end_date :str ,CFG :CFG ,*,y_col_prefer :str ="y_excess"):

    tds =get_trade_days (start_date =start_date ,end_date =end_date )
    trade_dates =[str (d )for d in tds ]

    root_a101 =Path (CFG .ALPHA101_DIR )
    root_a191 =Path (CFG .ALPHA191_DIR )
    root_jq =Path (CFG .JQF_DIR )
    root_pf =Path (CFG .PRIVATE_DIR )
    root_ind =Path (CFG .INDUSTRY_DIR )
    root_tgt =Path (CFG .TARGET_DIR )

    X_list =[]
    y_list =[]

    for d in tqdm (trade_dates ):

        tgt_path =root_tgt /f"{d}.parquet"
        ydf =_read_parquet_if_exists (tgt_path )
        if ydf .empty :
            continue 
        if "code"in ydf .columns :
            ydf =ydf .set_index ("code")

        ydf .index =ydf .index .astype (str )

        y_col =y_col_prefer if (y_col_prefer in ydf .columns )else ("y"if "y"in ydf .columns else None )
        if y_col is None :
            continue 


        a101 =_load_factor_day (root_a101 /f"{d}.parquet","a101_",date =d ,has_date_col =True )
        a191 =_load_factor_day (root_a191 /f"{d}.parquet","a191_",date =d ,has_date_col =True )
        jq =_load_factor_day (root_jq /f"{d}.parquet","jq_",date =d ,has_date_col =True )


        pf_path =root_pf /f"{d}.parquet"
        pf =_load_factor_day (pf_path ,"pf_",date =d ,has_date_col =False )


        Xraw =a101 .merge (a191 ,on ="code",how ="outer").merge (jq ,on ="code",how ="outer")
        if not pf .empty :
            Xraw =Xraw .merge (pf ,on ="code",how ="outer")

        Xraw =Xraw .dropna (subset =["code"]).drop_duplicates ("code",keep ="last")

        if Xraw .empty :
            continue 

        Xraw ["code"]=Xraw ["code"].astype (str )
        Xraw =Xraw .set_index ("code")


        ind_label =_get_industry_label_long (root_ind /f"{d}.parquet",Xraw .index )


        mcap_col =_pick_market_cap_column (Xraw .reset_index ())
        if mcap_col is None or (mcap_col not in Xraw .columns ):
            ln_mcap =pd .Series (np .nan ,index =Xraw .index ,dtype ="float64")
        else :
            mcap =pd .to_numeric (Xraw [mcap_col ],errors ="coerce").astype (float )
            mcap =mcap .clip (lower =1.0 )
            ln_mcap =np .log (mcap )


        Xraw [mcap_col ]=ln_mcap 


        Xnum =Xraw .apply (pd .to_numeric ,errors ="coerce")
        Xnum =_industry_median_fill (Xnum ,ind_label )


        Xw =_winsorize (Xnum ,q =0.01 )


        Xn =_neutralize_by_industry_and_size (
        Xw ,
        ind_label =ind_label ,
        ln_mcap =Xw [mcap_col ],
        exclude_cols =[mcap_col ],
        )


        Xz =_zscore (Xn )


        y_day =pd .DataFrame (index =ydf .index )
        y_day ["y"]=pd .to_numeric (ydf [y_col ],errors ="coerce").astype ("float32")


        for c in ["entry_date","exit_date"]:
            if c in ydf .columns :
                y_day [c ]=ydf [c ]
            else :
                y_day [c ]=np .nan 


        codes =Xz .index .intersection (y_day .index )
        if len (codes )==0 :
            continue 
        Xz =Xz .loc [codes ]
        y_day =y_day .loc [codes ]


        y_day =y_day .dropna (subset =["y"])
        if y_day .empty :
            continue 
        Xz =Xz .loc [y_day .index ]


        X_day =Xz .copy ()
        X_day .index .name ="code"
        X_day =X_day .reset_index ()
        X_day .insert (0 ,"date",pd .to_datetime (d ))
        X_day =X_day .set_index (["date","code"]).sort_index ()

        y_out =y_day .copy ()
        y_out .index .name ="code"
        y_out =y_out .reset_index ()
        y_out .insert (0 ,"date",pd .to_datetime (d ))
        y_out =y_out .set_index (["date","code"]).sort_index ()

        X_list .append (X_day )
        y_list .append (y_out )

    if not X_list :
        return pd .DataFrame (),pd .DataFrame ()

    X =pd .concat (X_list ,axis =0 ).sort_index ()
    y =pd .concat (y_list ,axis =0 ).sort_index ()


    idx =X .index .intersection (y .index )
    X =X .loc [idx ]
    y =y .loc [idx ]

    return X ,y 


def build_range_X_only (start_date :str ,end_date :str ,CFG :CFG ):

    tds =get_trade_days (start_date =start_date ,end_date =end_date )
    trade_dates =[str (d )for d in tds ]

    root_a101 =Path (CFG .ALPHA101_DIR )
    root_a191 =Path (CFG .ALPHA191_DIR )
    root_jq =Path (CFG .JQF_DIR )
    root_pf =Path (CFG .PRIVATE_DIR )
    root_ind =Path (CFG .INDUSTRY_DIR )

    X_list =[]

    for d in tqdm (trade_dates ,desc ="build_range_X_only"):

        a101 =_load_factor_day (root_a101 /f"{d}.parquet","a101_",date =d ,has_date_col =True )
        a191 =_load_factor_day (root_a191 /f"{d}.parquet","a191_",date =d ,has_date_col =True )
        jq =_load_factor_day (root_jq /f"{d}.parquet","jq_",date =d ,has_date_col =True )

        pf_path =root_pf /f"{d}.parquet"
        pf =_load_factor_day (pf_path ,"pf_",date =d ,has_date_col =False )


        Xraw =a101 .merge (a191 ,on ="code",how ="outer").merge (jq ,on ="code",how ="outer")
        if pf is not None and (not pf .empty ):
            Xraw =Xraw .merge (pf ,on ="code",how ="outer")

        Xraw =Xraw .dropna (subset =["code"]).drop_duplicates ("code",keep ="last")
        if Xraw .empty :
            continue 

        Xraw ["code"]=Xraw ["code"].astype (str )
        Xraw =Xraw .set_index ("code")


        ind_label =_get_industry_label_long (root_ind /f"{d}.parquet",Xraw .index )


        mcap_col =_pick_market_cap_column (Xraw .reset_index ())
        if mcap_col is None or (mcap_col not in Xraw .columns ):

            ln_mcap =pd .Series (np .nan ,index =Xraw .index ,dtype ="float64")

            mcap_col ="__ln_mcap__"
            Xraw [mcap_col ]=ln_mcap 
        else :
            mcap =pd .to_numeric (Xraw [mcap_col ],errors ="coerce").astype (float )
            mcap =mcap .clip (lower =1.0 )
            ln_mcap =np .log (mcap )
            Xraw [mcap_col ]=ln_mcap 


        Xnum =Xraw .apply (pd .to_numeric ,errors ="coerce")
        Xnum =_industry_median_fill (Xnum ,ind_label )


        Xw =_winsorize (Xnum ,q =0.01 )



        Xn =_neutralize_by_industry_and_size (
        Xw ,
        ind_label =ind_label ,
        ln_mcap =Xw [mcap_col ],
        exclude_cols =[mcap_col ],
        )


        Xz =_zscore (Xn )


        X_day =Xz .copy ()
        X_day .index .name ="code"
        X_day =X_day .reset_index ()
        X_day .insert (0 ,"date",pd .to_datetime (d ))
        X_day =X_day .set_index (["date","code"]).sort_index ()

        X_list .append (X_day )

    if not X_list :
        return pd .DataFrame ()

    X =pd .concat (X_list ,axis =0 ).sort_index ()
    return X 
