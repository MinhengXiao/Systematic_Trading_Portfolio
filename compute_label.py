
import numpy as np 
import pandas as pd 
from pathlib import Path 
import datetime as dt 

def _to_datestr (d ):
    return pd .to_datetime (d ).strftime ("%Y-%m-%d")

def _infer_entry_exit_by_jq_trade_calendar (
t_date ,
holding_period :int ,
label_mode :str ,
*,
buffer_extra :int =10 ,
max_expand :int =6 ,
):

    t =pd .to_datetime (t_date ).date ()


    if label_mode =="open_next":

        need_offset =holding_period 
    elif label_mode =="close_next":

        need_offset =holding_period +1 
    elif label_mode =="close_today":

        need_offset =holding_period 
    else :
        raise ValueError ("label_mode must be one of {'open_next','close_next','close_today'}")


    end =t +dt .timedelta (days =max (30 ,(need_offset +buffer_extra )*3 ))
    days =None 
    for _ in range (max_expand ):
        days =get_trade_days (start_date =t ,end_date =end )
        if days is not None and len (days )>=(need_offset +1 ):
            break 
        end =end +dt .timedelta (days =30 )

    if days is None or len (days )<(need_offset +1 ):
        raise ValueError (f"Insufficient future trading days: t={t}, need_offset={need_offset}, got={0 if days is None else len(days)}")

    days =[pd .to_datetime (x ).date ()for x in days ]
    d_signal =days [0 ]
    if d_signal !=t :

        raise ValueError (f"{t} is not a trading day (get_trade_days returned first day as {d_signal}).Please pass a trading day as the signal date.")

    if label_mode =="open_next":
        d_entry =days [1 ]
        d_exit =days [holding_period ]
        entry_px_field ="open"
        entry_for_index_field ="open"
    elif label_mode =="close_next":
        d_entry =days [1 ]
        d_exit =days [holding_period +1 ]
        entry_px_field ="close"
        entry_for_index_field ="close"
    else :
        d_entry =days [0 ]
        d_exit =days [holding_period ]
        entry_px_field ="close"
        entry_for_index_field ="close"

    return (
    _to_datestr (d_signal ),
    _to_datestr (d_entry ),
    _to_datestr (d_exit ),
    entry_px_field ,
    entry_for_index_field ,
    )


def forward_return_local (
date :str ,
holding_period :int =3 ,
label_mode :str ="open_next",
*,
local_root :str ="./local_data",
bench_code :str ="000905.XSHG",
):

    if holding_period <1 :
        raise ValueError ("holding_period must be >= 1")


    d_signal ,d_entry ,d_exit ,entry_px_field ,entry_for_index_field =_infer_entry_exit_by_jq_trade_calendar (
    date ,holding_period ,label_mode 
    )

    root =Path (local_root )
    stock_dir =root /"get_price_stock"/"daily"
    index_dir =root /"get_price_index"/"daily"


    p_entry =stock_dir /f"{d_entry}.parquet"
    p_exit =stock_dir /f"{d_exit}.parquet"
    if (not p_entry .exists ())or (not p_exit .exists ()):
        raise FileNotFoundError (f"Missingstock daily bar files: {p_entry} or {p_exit}")

    df_e =pd .read_parquet (p_entry ).copy ()
    df_x =pd .read_parquet (p_exit ).copy ()

    if "code"not in df_e .columns or "code"not in df_x .columns :
        raise ValueError ("localstock daily bar parquet Missing code column (expected code/open/close/...)")

    df_e ["code"]=df_e ["code"].astype (str )
    df_x ["code"]=df_x ["code"].astype (str )
    df_e =df_e .set_index ("code")
    df_x =df_x .set_index ("code")

    if entry_px_field not in df_e .columns :
        raise ValueError (f"entry fileMissingfield {entry_px_field}: {p_entry}")
    if "close"not in df_x .columns :
        raise ValueError (f"exit fileMissingfield close: {p_exit}")

    entry_price =pd .to_numeric (df_e [entry_px_field ],errors ="coerce")
    exit_price =pd .to_numeric (df_x ["close"],errors ="coerce")

    with np .errstate (divide ="ignore",invalid ="ignore"):
        y =np .log (exit_price /entry_price )
    y =y .replace ([np .inf ,-np .inf ],np .nan )

    out =pd .DataFrame ({"y":y })


    p_ie =index_dir /f"{d_entry}.parquet"
    p_ix =index_dir /f"{d_exit}.parquet"
    if (not p_ie .exists ())or (not p_ix .exists ()):
        raise FileNotFoundError (f"Missingindex daily bar files: {p_ie} or {p_ix}")

    idx_e =pd .read_parquet (p_ie ).copy ()
    idx_x =pd .read_parquet (p_ix ).copy ()

    if "code"in idx_e .columns :
        idx_e ["code"]=idx_e ["code"].astype (str )
        idx_e =idx_e [idx_e ["code"]==str (bench_code )]
    if "code"in idx_x .columns :
        idx_x ["code"]=idx_x ["code"].astype (str )
        idx_x =idx_x [idx_x ["code"]==str (bench_code )]

    if idx_e .empty or idx_x .empty :
        raise ValueError (f"Benchmark index not found in index files bench_code={bench_code}: {p_ie} or {p_ix}")

    if entry_for_index_field not in idx_e .columns :
        raise ValueError (f"Index entry fileMissingfield {entry_for_index_field}: {p_ie}")
    if "close"not in idx_x .columns :
        raise ValueError (f"Index exit fileMissingfield close: {p_ix}")

    idx_entry =float (pd .to_numeric (idx_e .iloc [0 ][entry_for_index_field ],errors ="coerce"))
    idx_exit =float (pd .to_numeric (idx_x .iloc [0 ]["close"],errors ="coerce"))

    if not np .isfinite (idx_entry )or not np .isfinite (idx_exit )or idx_entry <=0 or idx_exit <=0 :
        idx_y =np .nan 
    else :
        idx_y =float (np .log (idx_exit /idx_entry ))

    out ["y_excess"]=out ["y"]-idx_y 


    out ["signal_date"]=d_signal 
    out ["entry_date"]=d_entry 
    out ["exit_date"]=d_exit 
    out ["k"]=int (holding_period )
    out ["label_mode"]=label_mode 
    out ["bench"]=bench_code 

    return out 
