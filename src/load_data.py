
import pandas as pd
import numpy as np

def load_data():
    df_info = pd.read_csv('/Users/mattdevor/galvanize/capstone_2/data/seinfeld-chronicles/episode_info.csv', index_col=0)
    df_scripts = pd.read_csv('/Users/mattdevor/galvanize/capstone_2/data/seinfeld-chronicles/scripts.csv', index_col=0)

    # Fix season 1 episode 0 issue
    df_scripts.EpisodeNo = np.where(df_scripts.SEID =='S01E00', 0.0, df_scripts.EpisodeNo)

    # drop NAs
    df_scripts = df_scripts.dropna()

    df_scripts = df_scripts.reset_index(drop=True)
    
    return df_info, df_scripts

def agg_dialogue_by_episode(df_scripts, df_info):
    df_scripts = df_scripts.copy()
    df_info = df_info.copy()

    all_documents = []
    df_cols = ['Dialogue', 'Lines_of_Dialogue', 'SEID', 'Season', 'Episode']
    df_new = pd.DataFrame(columns = df_cols)
    
    index = 0
    for SEID in df_scripts['SEID'].unique():
        dialogue = ' '.join(df_scripts[df_scripts['SEID']==SEID]['Dialogue'].to_list())
        lines_of_dialogue = int(df_scripts[df_scripts['SEID']==SEID]['EpisodeNo'].count())
        season = df_scripts[df_scripts['SEID']==SEID]['Season'].unique()[0]
        episode = df_scripts[df_scripts['SEID']==SEID]['EpisodeNo'].unique()[0]
        
        df_new.loc[index] = [dialogue, lines_of_dialogue, SEID, season, episode]
        
        index += 1

    merged = pd.merge(df_new, df_info.iloc[:,2:], on=['SEID']).reset_index(drop=True)

    merged['Lines_of_Dialogue'] = merged['Lines_of_Dialogue'].astype(int)

    return merged

def get_jerry_df(df):
    df = df.copy()
    
    df.Character = df.Character.astype(str)
    df.Dialogue = df.Dialogue.astype(str)
    
    df = df[["Character","Dialogue"]].copy()
    
    char = 'JERRY'
    
    idx_first_line = df[df.Character==char].index[0]
    
    
    # lists to create new df - this "should" be much faster than appending to df at each row
    # Get first line and the line after
    q_char_lst = [df.iloc[idx_first_line,0]]
    q_dialogue_lst = [df.iloc[idx_first_line,1]]
    a_char_lst = [df.iloc[idx_first_line+1,0]]
    a_dialogue_lst = [df.iloc[idx_first_line+1,1]]
    
    for index, row in df.iloc[idx_first_line + 2:, :].iterrows(): 
        index_before = index - 1
        
        if row.Character != char:
            continue
            
        q_char_lst.append(df.iloc[index_before,0])
        q_dialogue_lst.append(df.iloc[index_before,1])
        a_char_lst.append(df.iloc[index,0])
        a_dialogue_lst.append(df.iloc[index,1])

    new_df_cols = ["q_char", "q_dialogue", "a_char", "a_dialogue"]
    df_jerry = pd.DataFrame(np.column_stack([q_char_lst, q_dialogue_lst, a_char_lst, a_dialogue_lst]), 
                               columns=new_df_cols)
    return df_jerry

def parse_data_save_txt(df):
    outF = open("data/jerry_q_a.txt", "w")
    for index, row in df.iterrows():
        # write line to output file
        outF.write(row.q_dialogue)
        outF.write("\t")
        outF.write(row.a_dialogue)
        outF.write("\n")
    outF.close()

if __name__ == "__main__":
    # Import Dataset
    df_info, df_scripts = load_data()
    df_docs_by_ep = agg_dialogue_by_episode(df_scripts, df_info)

    # Parse out Jerry Q-A pairs
    df_get_jerry = get_jerry_df(df_scripts)

    # Create .txt file for Jerry Q-A pairs
    parse_data_save_txt(df_get_jerry)