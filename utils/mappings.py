VIEW_MAP_RAW_TO_IDX = { -1:0, 1:1, 2:2, 3:3, 4:4, 5:5 }
VIEW_IDX_TO_NAME    = [
    'uncertain','front','rear','side','front-side','rear-side'
]

def build_year_index(years):
    sorted_years = sorted(sorted(set(years)))
    year_to_idx = {y:i for i,y in enumerate(sorted_years)}
    idx_to_year = {i:y for y,i in year_to_idx.items()}
    return year_to_idx, idx_to_year

TYPE_ID_TO_IDX = {i:i-1 for i in range(1,13)}
TYPE_IDX_TO_ID = {v:k for k,v in TYPE_ID_TO_IDX.items()}